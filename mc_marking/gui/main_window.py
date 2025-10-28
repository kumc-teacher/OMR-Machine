"""PyQt6 main window driving the MC Marking workflow."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PyQt6.QtCore import QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QInputDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtWidgets import QRubberBand

import pytesseract
from pytesseract import TesseractNotFoundError

from mc_marking.models.answer_sheet import AnswerKey, CellResult, PageAnswer, PageResult, TableExtraction
from mc_marking.services.answer_key_service import normalize_answer_key
from mc_marking.services.image_loader import LoadedPage, load_pages
from mc_marking.services.marking_service import evaluate_page
from mc_marking.services.ocr_service import recognise_table_cells
from mc_marking.services.page_processor import build_page_result
from mc_marking.services.table_detection import detect_table, detect_tables
from mc_marking.utils.image_utils import BoundingBox
from mc_marking.utils.settings import AppSettings, load_settings, save_settings

SUPPORTED_FILTER = "PDF or Images (*.pdf *.png *.jpg *.jpeg *.bmp *.tif *.tiff)"


class ImageCanvas(QLabel):
    """Image display widget that supports manual rectangular selection."""

    selection_changed = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(True)
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._origin: Optional[QPoint] = None
        self._selection: Optional[QRect] = None
        self._image: Optional[np.ndarray] = None
        self._original_pixmap: Optional[QPixmap] = None

    def set_image(self, image: np.ndarray) -> None:
        self._image = image
        self._selection = None
        self._rubber_band.hide()
        qimage = _np_to_qimage(image)
        pixmap = QPixmap.fromImage(qimage)
        self._original_pixmap = pixmap
        self.setPixmap(pixmap)

    def clear(self) -> None:
        self._image = None
        self._selection = None
        self._rubber_band.hide()
        self.setPixmap(QPixmap())
        self._original_pixmap = None

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._image is None:
            return
        self._origin = event.position().toPoint()
        self._rubber_band.setGeometry(QRect(self._origin, QSize()))
        self._rubber_band.show()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._origin is None or self._image is None:
            return
        current = event.position().toPoint()
        rect = QRect(self._origin, current).normalized()
        self._rubber_band.setGeometry(rect)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._origin is None or self._image is None:
            return
        current = event.position().toPoint()
        self._selection = QRect(self._origin, current).normalized()
        self.selection_changed.emit(self.get_selection_box())
        self._origin = None

    def get_selection_box(self) -> Optional[BoundingBox]:
        if self._image is None or self._selection is None or self._selection.isNull():
            return None
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        label_width = max(1, self.width())
        label_height = max(1, self.height())
        scale_x = pixmap_width / label_width
        scale_y = pixmap_height / label_height
        rect = self._selection
        x = int(rect.x() * scale_x)
        y = int(rect.y() * scale_y)
        width = int(rect.width() * scale_x)
        height = int(rect.height() * scale_y)
        width = max(1, width)
        height = max(1, height)
        return BoundingBox(x=x, y=y, width=width, height=height)

    def original_pixmap(self) -> Optional[QPixmap]:
        return self._original_pixmap


class MainWindow(QMainWindow):
    """Top-level window orchestrating answer key capture and marking."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MC Marking")
        self.resize(1280, 820)

        self.settings: AppSettings = load_settings()
        self._apply_runtime_paths()
        self.baseline_density: Optional[float] = self.settings.baseline_ink_density
        self.row_labels_override: List[str] = list(self.settings.row_labels)

        self.answer_key: Optional[AnswerKey] = None
        self.answer_extractions: List[TableExtraction] = []
        self.answer_page: Optional[LoadedPage] = None
        self.answer_cells: List[List[CellResult]] = []
        self.page_results: List[PageResult] = []

        self.canvas = ImageCanvas()
        self.canvas.selection_changed.connect(self._on_selection_changed)

        self.answer_table = QTableWidget(0, 2)
        self.answer_table.setHorizontalHeaderLabels(["Question", "Answer"])
        self.answer_table.horizontalHeader().setStretchLastSection(True)

        self.answer_log = QTextEdit()
        self.answer_log.setReadOnly(True)

        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(["Source", "Correct", "Incorrect", "Unanswered", "Total"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.itemSelectionChanged.connect(self._on_page_result_selected)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self.student_answers_table = QTableWidget(0, 4)
        self.student_answers_table.setHorizontalHeaderLabels(["Question", "Detected", "Confidence", "Status"])
        self.student_answers_table.horizontalHeader().setStretchLastSection(True)
        self.student_answers_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.student_answers_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.student_answers_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.reset_zoom_button.clicked.connect(lambda: self.zoom_slider.setValue(100))
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom"))
        zoom_row.addWidget(self.zoom_slider)
        zoom_row.addWidget(self.reset_zoom_button)

        self.calibrate_button = QPushButton("Calibrate Blank Sheet")
        self.calibrate_button.clicked.connect(self._calibrate_blank_sheet)

        self.configure_poppler_button = QPushButton("Set Poppler Path")
        self.configure_poppler_button.clicked.connect(self._configure_poppler_path)

        self.configure_tesseract_button = QPushButton("Set Tesseract Path")
        self.configure_tesseract_button.clicked.connect(self._configure_tesseract_path)

        self.load_answer_button = QPushButton("Load Answer Key")
        self.load_answer_button.clicked.connect(self._load_answer_key)

        self.use_selection_button = QPushButton("Use Selection")
        self.use_selection_button.setEnabled(False)
        self.use_selection_button.clicked.connect(self._apply_manual_selection)

        self.process_button = QPushButton("Process Answer Sheets")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self._process_answer_sheets)

        self.configure_layout_button = QPushButton("Configure Layout")
        self.configure_layout_button.clicked.connect(self._configure_row_labels)

        self.export_button = QPushButton("Export Results")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_results)

        left_panel = QVBoxLayout()
        left_panel.addWidget(self.configure_poppler_button)
        left_panel.addWidget(self.configure_tesseract_button)
        left_panel.addWidget(self.load_answer_button)
        left_panel.addWidget(self.use_selection_button)
        left_panel.addWidget(self.process_button)
        left_panel.addWidget(self.configure_layout_button)
        left_panel.addWidget(self.export_button)
        left_panel.addWidget(self.calibrate_button)

        left_panel.addWidget(_build_group_box("Answer Key", self.answer_table))
        left_panel.addWidget(_build_group_box("Recognition Log", self.answer_log))
        left_panel_widget = QWidget()
        left_panel_widget.setLayout(left_panel)

        central_layout = QHBoxLayout()
        central_layout.addWidget(left_panel_widget, 1)

        preview_panel = QVBoxLayout()
        preview_panel.setContentsMargins(0, 0, 0, 0)
        preview_panel.addWidget(_build_group_box("Preview", self.canvas))
        preview_panel.addLayout(zoom_row)
        preview_widget = QWidget()
        preview_widget.setLayout(preview_panel)
        central_layout.addWidget(preview_widget, 1)

        right_panel = QVBoxLayout()
        right_panel.addWidget(_build_group_box("Page Results", self.results_table))
        right_panel.addWidget(_build_group_box("Student Answers", self.student_answers_table))
        right_panel_widget = QWidget()
        right_panel_widget.setLayout(right_panel)
        central_layout.addWidget(right_panel_widget, 1)

        central_widget = QWidget()
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _load_answer_key(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, "Select answer key", str(Path.home()), SUPPORTED_FILTER)
        if not file_name:
            return
        try:
            pages = load_pages([Path(file_name)], poppler_path=self.settings.poppler_path)
        except Exception as exc:  # pragma: no cover - GUI feedback
            QMessageBox.critical(self, "Load Error", f"Failed to load file: {exc}")
            return
        if not pages:
            QMessageBox.warning(self, "No Pages", "No pages were extracted from the selected file.")
            return

        page = pages[0]
        self.answer_page = page
        self.canvas.set_image(page.image)
        self.status_bar.showMessage("Detecting tables in answer key...")
        extractions = detect_tables(page.image, page.page_index, str(page.source))
        if not extractions:
            self.answer_log.setPlainText(
                "Automatic detection failed. Draw a rectangle around the tables and choose 'Use Selection'."
            )
            self.use_selection_button.setEnabled(True)
            self.process_button.setEnabled(False)
            self.status_bar.showMessage("Detection failed; awaiting manual selection.")
            return
        self._finalize_answer_key(page, extractions)

    def _apply_manual_selection(self) -> None:
        if self.answer_page is None or self.answer_page.image is None:
            return
        selection = self.canvas.get_selection_box()
        if selection is None:
            QMessageBox.information(self, "No Selection", "Draw a rectangle over the table first.")
            return
        self.status_bar.showMessage("Running detection on selected area...")
        extractions = detect_tables(
            self.answer_page.image,
            self.answer_page.page_index,
            str(self.answer_page.source),
            roi=selection,
        )
        if not extractions:
            self.status_bar.showMessage("Detection still failed. Adjust the selection and try again.")
            return
        self._finalize_answer_key(self.answer_page, extractions)

    def _finalize_answer_key(self, page: LoadedPage, extractions: List[TableExtraction]) -> None:
        tables_cells: List[List[CellResult]] = []
        try:
            for extraction in extractions:
                ocr_results = recognise_table_cells(page.image, extraction)
                extraction.cells = ocr_results
                tables_cells.append(ocr_results)
        except TesseractNotFoundError:
            QMessageBox.critical(
                self,
                "Tesseract Not Found",
                "Tesseract OCR executable could not be located. Install Tesseract or choose it via 'Set Tesseract Path'.",
            )
            self.status_bar.showMessage("Set the Tesseract path before proceeding.", 5000)
            return
        answer_key = normalize_answer_key(
            extractions,
            tables_cells,
            baseline_density=self.baseline_density,
            row_labels_override=self.row_labels_override,
        )
        if not answer_key.rows:
            self.answer_log.setPlainText(
                "No valid rows were recognized. Ensure the table has question numbers in the first column."
            )
            self.status_bar.showMessage("Recognition incomplete.")
            return
        self.answer_key = answer_key
        self.answer_extractions = list(extractions)
        self.answer_cells = tables_cells
        self.page_results = []
        self.results_table.setRowCount(0)
        self.student_answers_table.setRowCount(0)
        self.use_selection_button.setEnabled(True)
        self.process_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self._populate_answer_table(answer_key)
        log_lines = [f"Q{question}: {answer}" for question, answer in sorted(answer_key.rows.items())]
        self.answer_log.setPlainText("\n".join(log_lines))
        self.status_bar.showMessage(f"Captured answer key from {len(extractions)} table(s).", 5000)

    def _process_answer_sheets(self) -> None:
        if self.answer_key is None:
            QMessageBox.information(self, "Answer Key Needed", "Load an answer key before processing sheets.")
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Select answer sheets", str(Path.home()), SUPPORTED_FILTER)
        if not files:
            return
        try:
            pages = load_pages((Path(file) for file in files), poppler_path=self.settings.poppler_path)
        except Exception as exc:  # pragma: no cover - GUI feedback
            QMessageBox.critical(self, "Load Error", f"Failed to load pages: {exc}")
            return
        results: List[PageResult] = []
        failures: List[str] = []
        for page in pages:
            extractions = detect_tables(page.image, page.page_index, str(page.source))
            if not extractions and self.answer_extractions:
                extractions = []
                for reference in self.answer_extractions:
                    if reference.bounding_box is None:
                        continue
                    fallback = detect_table(
                        page.image,
                        page.page_index,
                        str(page.source),
                        roi=reference.bounding_box,
                    )
                    if fallback is not None and fallback.cells:
                        extractions.append(fallback)
            if not extractions:
                failures.append(f"{page.source.name} (page {page.page_index + 1})")
                continue
            tables_cells: List[List[CellResult]] = []
            try:
                for extraction in extractions:
                    ocr_cells = recognise_table_cells(page.image, extraction)
                    extraction.cells = ocr_cells
                    tables_cells.append(ocr_cells)
            except TesseractNotFoundError:
                QMessageBox.critical(
                    self,
                    "Tesseract Not Found",
                    "Processing stopped because Tesseract is unavailable. Install it or set the executable path.",
                )
                self.status_bar.showMessage("Set the Tesseract path before processing.", 5000)
                return
            page_result = build_page_result(
                extractions,
                tables_cells,
                baseline_density=self.baseline_density,
                row_labels_override=self.row_labels_override,
            )
            evaluated = evaluate_page(page_result, self.answer_key)
            results.append(evaluated)
        self.page_results = results
        self._populate_results_table()
        if self.page_results:
            self.results_table.selectRow(0)
        else:
            self.student_answers_table.setRowCount(0)
        self.export_button.setEnabled(bool(self.page_results))
        if failures:
            self.answer_log.append("\nUnprocessed pages:\n" + "\n".join(failures))
        self.status_bar.showMessage(f"Processed {len(results)} pages.", 5000)

    def _populate_answer_table(self, answer_key: AnswerKey) -> None:
        rows = sorted(answer_key.rows.items())
        self.answer_table.setRowCount(len(rows))
        for row_idx, (question, answer) in enumerate(rows):
            self.answer_table.setItem(row_idx, 0, QTableWidgetItem(str(question)))
            self.answer_table.setItem(row_idx, 1, QTableWidgetItem(answer))

    def _on_zoom_changed(self, value: int) -> None:
        base_pixmap = self.canvas.original_pixmap()
        if base_pixmap is None or base_pixmap.isNull():
            return
        scale_factor = value / 100.0
        scaled = base_pixmap.scaled(
            base_pixmap.size() * scale_factor,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.canvas.setPixmap(scaled)

    def _populate_results_table(self) -> None:
        self.results_table.blockSignals(True)
        self.results_table.setRowCount(len(self.page_results))
        for row_idx, result in enumerate(self.page_results):
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(f"{result.source_path.name} (p{result.page_index + 1})"))
            self.results_table.setItem(row_idx, 1, QTableWidgetItem(str(result.correct_count)))
            self.results_table.setItem(row_idx, 2, QTableWidgetItem(str(result.incorrect_count)))
            self.results_table.setItem(row_idx, 3, QTableWidgetItem(str(result.unanswered_count)))
            self.results_table.setItem(row_idx, 4, QTableWidgetItem(str(result.total_questions)))
        self.results_table.blockSignals(False)

    def _on_page_result_selected(self) -> None:
        selection_model = self.results_table.selectionModel()
        if selection_model is None:
            return
        selected_rows = selection_model.selectedRows()
        if not selected_rows:
            self.student_answers_table.setRowCount(0)
            return
        row = selected_rows[0].row()
        if 0 <= row < len(self.page_results):
            self._populate_student_answers(self.page_results[row])

    def _populate_student_answers(self, page_result: PageResult) -> None:
        answers = page_result.answers
        self.student_answers_table.setRowCount(len(answers))
        for row_idx, answer in enumerate(answers):
            self.student_answers_table.setItem(row_idx, 0, QTableWidgetItem(str(answer.question)))
            self.student_answers_table.setItem(row_idx, 1, QTableWidgetItem(answer.extracted))
            self.student_answers_table.setItem(row_idx, 2, QTableWidgetItem(f"{answer.confidence:.2f}"))
            expected = self.answer_key.answer_for(answer.question) if self.answer_key else ""
            if answer.is_correct is True:
                status = "Correct"
            elif answer.is_correct is False:
                status = f"Incorrect (expected {expected})" if expected else "Incorrect"
            else:
                status = "Unanswered" if not answer.extracted else "Review"
            self.student_answers_table.setItem(row_idx, 3, QTableWidgetItem(status))

    def _calibrate_blank_sheet(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, "Select blank answer sheet", str(Path.home()), SUPPORTED_FILTER)
        if not file_name:
            return
        try:
            pages = load_pages([Path(file_name)], poppler_path=self.settings.poppler_path)
        except Exception as exc:  # pragma: no cover - GUI feedback
            QMessageBox.critical(self, "Calibration Error", f"Failed to load calibration sheet: {exc}")
            return
        if not pages:
            QMessageBox.warning(self, "Calibration", "No pages found in calibration file.")
            return
        page = pages[0]
        extractions = detect_tables(page.image, page.page_index, str(page.source))
        if not extractions:
            QMessageBox.warning(self, "Calibration", "No tables detected on the calibration sheet.")
            return
        tables_cells: List[List[CellResult]] = []
        for extraction in extractions:
            try:
                ocr_cells = recognise_table_cells(page.image, extraction)
            except TesseractNotFoundError:
                QMessageBox.critical(
                    self,
                    "Tesseract Not Found",
                    "Calibration stopped because Tesseract is unavailable. Install it or set the executable path.",
                )
                return
            extraction.cells = ocr_cells
            tables_cells.append(ocr_cells)
        averages = []
        for cells in tables_cells:
            if not cells:
                continue
            avg_density = sum(cell.ink_density for cell in cells) / len(cells)
            averages.append(avg_density)
        if not averages:
            QMessageBox.information(self, "Calibration", "No cell data available for calibration.")
            return
        baseline = sum(averages) / len(averages)
        self.baseline_density = baseline
        self.settings.baseline_ink_density = baseline
        save_settings(self.settings)
        QMessageBox.information(self, "Calibration", f"Baseline ink density recorded: {baseline:.4f}")
        self.status_bar.showMessage("Calibration baseline saved", 5000)

    def _configure_row_labels(self) -> None:
        current = ", ".join(self.row_labels_override) if self.row_labels_override else "A, B, C, D"
        text, ok = QInputDialog.getText(
            self,
            "Configure Choice Labels",
            "Enter choice labels in order (comma-separated):",
            text=current,
        )
        if not ok:
            return
        normalized = text.replace(";", ",").replace("\n", ",")
        labels = [part.strip().upper() for part in normalized.split(",") if part.strip()]
        if len(labels) <= 1:
            labels = [part.strip().upper() for part in text.split() if part.strip()]
        if not labels:
            QMessageBox.warning(self, "Invalid Labels", "Please provide at least one label (e.g., A, B, C, D).")
            return
        self.row_labels_override = labels
        self.settings.row_labels = labels
        save_settings(self.settings)
        self.status_bar.showMessage("Choice labels updated", 5000)
        QMessageBox.information(
            self,
            "Layout Updated",
            "Choice labels saved. Reload the answer key to apply them if needed.",
        )

    def _export_results(self) -> None:
        if not self.page_results:
            QMessageBox.information(self, "Export Results", "Process answer sheets before exporting.")
            return
        default_name = str(Path.home() / "mc_marking_results.csv")
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            default_name,
            "CSV Files (*.csv)",
        )
        if not file_name:
            return

        export_path = Path(file_name)
        if export_path.suffix.lower() != ".csv":
            export_path = export_path.with_suffix(".csv")

        if self.answer_key and self.answer_key.rows:
            question_numbers = set(self.answer_key.rows.keys())
        else:
            question_numbers = set()
        for result in self.page_results:
            for answer in result.answers:
                question_numbers.add(answer.question)
        ordered_questions = sorted(question_numbers)

        header = [
            "Source",
            "Page",
            "Correct",
            "Incorrect",
            "Unanswered",
            "Total",
        ]
        for question in ordered_questions:
            header.append(f"Q{question} Answer")
            header.append(f"Q{question} Status")

        try:
            with open(export_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)

                for result in self.page_results:
                    answer_map: Dict[int, PageAnswer] = {answer.question: answer for answer in result.answers}
                    row = [
                        result.source_path.name,
                        result.page_index + 1,
                        result.correct_count,
                        result.incorrect_count,
                        result.unanswered_count,
                        result.total_questions,
                    ]
                    for question in ordered_questions:
                        answer = answer_map.get(question)
                        detected = answer.extracted if answer else ""
                        status: str
                        if answer is None or not answer.extracted:
                            status = "Unanswered"
                        elif answer.is_correct is True:
                            status = "Correct"
                        elif answer.is_correct is False:
                            status = "Incorrect"
                        else:
                            status = "Review"
                        row.extend([detected, status])
                    writer.writerow(row)
        except OSError as exc:
            QMessageBox.critical(self, "Export Failed", f"Could not write file:\n{exc}")
            return

        self.status_bar.showMessage(f"Exported results to {export_path}", 5000)
        QMessageBox.information(self, "Export Complete", f"Student results exported to {export_path}.")

    def _on_selection_changed(self, selection: Optional[BoundingBox]) -> None:
        self.use_selection_button.setEnabled(selection is not None)
        if selection is not None:
            self.status_bar.showMessage(
                f"Selection ({selection.width}x{selection.height}) at ({selection.x}, {selection.y})",
                3000,
            )

    def _configure_poppler_path(self) -> None:
        initial_dir = self.settings.poppler_path or str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Select Poppler bin directory", initial_dir)
        if not directory:
            return
        self.settings.poppler_path = directory
        save_settings(self.settings)
        self._apply_runtime_paths()
        self.status_bar.showMessage(f"Poppler path set to {directory}", 5000)

    def _configure_tesseract_path(self) -> None:
        initial_dir = self.settings.tesseract_path or str(Path.home())
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select tesseract executable",
            initial_dir,
            "Tesseract Executable (tesseract.exe)",
        )
        if not file_name:
            return
        self.settings.tesseract_path = file_name
        save_settings(self.settings)
        self._apply_runtime_paths()
        self.status_bar.showMessage(f"Tesseract path set to {file_name}", 5000)

    def _apply_runtime_paths(self) -> None:
        if self.settings.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.settings.tesseract_path
        else:
            pytesseract.pytesseract.tesseract_cmd = "tesseract"


def _build_group_box(title: str, widget: QWidget) -> QGroupBox:
    box = QGroupBox(title)
    layout = QVBoxLayout()
    layout.addWidget(widget)
    box.setLayout(layout)
    return box


def _np_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (h, w, 3)")
    height, width, _ = image.shape
    bytes_per_line = 3 * width
    return QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()


def run_app() -> None:
    """Launch the Qt event loop."""
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
