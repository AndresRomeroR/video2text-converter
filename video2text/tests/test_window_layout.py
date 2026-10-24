"""Opt-in checks against real Tk geometry; no screenshots or model loading."""

import os
import sys
import tkinter as tk
import tkinter.font as tkfont
import unittest
from pathlib import Path
from tkinter import ttk


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import video2text


@unittest.skipUnless(
    os.environ.get("VIDEO2TEXT_RUN_UI_TESTS") == "1",
    "Pruebas de ventana desactivadas; define VIDEO2TEXT_RUN_UI_TESTS=1.",
)
class WindowLayoutTests(unittest.TestCase):
    def create_app(self, scale, width=650, height=480):
        class ScaledApp(video2text.Video2TextApp):
            def _apply_theme(self):
                self.tk.call("tk", "scaling", scale * 96.0 / 72.0)
                self._scale = scale
                super()._apply_theme()

            def _detect_acceleration(self):
                pass  # Layout checks must never import Torch or load models.

        app = ScaledApp()
        self.addCleanup(self.close_app, app)
        app.geometry(f"{app._px(width)}x{app._px(height)}+0+0")
        app.update()
        app.update_idletasks()
        return app

    @staticmethod
    def close_app(app):
        for timer in app.tk.splitlist(app.tk.call("after", "info")):
            app.after_cancel(timer)
        app.close()

    def assert_inside_window(self, app, widget, name):
        left = widget.winfo_rootx() - app.winfo_rootx()
        top = widget.winfo_rooty() - app.winfo_rooty()
        right, bottom = left + widget.winfo_width(), top + widget.winfo_height()
        self.assertTrue(widget.winfo_ismapped(), f"{name} is not mapped")
        self.assertGreaterEqual(left, 0, f"{name} left={left}")
        self.assertGreaterEqual(top, 0, f"{name} top={top}")
        self.assertLessEqual(right, app.winfo_width(), f"{name} right={right}, window={app.winfo_width()}")
        self.assertLessEqual(bottom, app.winfo_height(), f"{name} bottom={bottom}, window={app.winfo_height()}")
        child = widget
        while child is not app:
            parent = child.master
            relative_x = child.winfo_rootx() - parent.winfo_rootx()
            relative_y = child.winfo_rooty() - parent.winfo_rooty()
            self.assertGreaterEqual(relative_x, 0, f"{name} clipped at parent left")
            self.assertGreaterEqual(relative_y, 0, f"{name} clipped at parent top")
            self.assertLessEqual(relative_x + child.winfo_width(), parent.winfo_width(),
                                 f"{name} clipped horizontally by {parent}")
            self.assertLessEqual(relative_y + child.winfo_height(), parent.winfo_height(),
                                 f"{name} clipped vertically by {parent}")
            child = parent

    def assert_main_controls_visible(self, app):
        for name in ("btn_process", "btn_stop", "btn_retry", "lbl_summary", "tabs", "file_tree"):
            self.assert_inside_window(app, getattr(app, name), name)
        if app._compact:
            self.assertFalse(app.txt_detail.winfo_ismapped())
            self.assertFalse(app.lbl_drop.winfo_ismapped())
            self.assertIn("arrastra", app.tabs.tab(0, "text"))
        else:
            for name in ("txt_detail", "btn_open_folder", "lbl_drop"):
                self.assert_inside_window(app, getattr(app, name), name)

    def assert_two_rows_visible(self, app):
        # bbox() reports actual exposed rows, including any clipped row height.
        app.file_tree.yview_moveto(0)
        app.update_idletasks()
        row_height = int(ttk.Style(app).lookup("Treeview", "rowheight"))
        font_height = tkfont.nametofont("TkDefaultFont", root=app).metrics("linespace")
        self.assertGreaterEqual(row_height, font_height)
        ids = app.file_tree.get_children()
        for index in range(2):
            bbox = app.file_tree.bbox(ids[index])
            self.assertTrue(bbox, f"Row {index+1} hidden; tree height={app.file_tree.winfo_height()}, rowheight={row_height}")
            self.assertLessEqual(bbox[1] + bbox[3], app.file_tree.winfo_height(),
                                 f"Row {index+1} clipped; bbox={bbox}, tree height={app.file_tree.winfo_height()}")

    def check_minimum_window(self, scale):
        app = self.create_app(scale)
        items, _ = app._batch.add([Path("uno.mp3"), Path("dos.mp4"), Path("tres.wav")])
        for item in items:
            app._render_item(item)
        app._refresh_summary()
        app.update()
        self.assert_main_controls_visible(app)
        self.assert_two_rows_visible(app)

    def test_minimum_window_at_100_percent(self):
        self.check_minimum_window(1.0)

    def test_minimum_window_at_150_percent(self):
        self.check_minimum_window(1.5)

    def check_long_list(self, scale, expanded=False):
        app = self.create_app(scale, 920, 700) if expanded else self.create_app(scale)
        items, _ = app._batch.add(
            Path("Carpeta con nombre extenso") / f"{index:03d}_{'nombre_largo_' * 12}.mp4"
            for index in range(100)
        )
        for item in items:
            app._render_item(item)
        app._refresh_summary()
        app.update()
        heights = {name: getattr(app, name).winfo_height() for name in
                   ("tabs", "file_tree", "txt_detail", "btn_process", "lbl_summary")}
        items[0].error = "No se pudo procesar el archivo. " * 100
        app.file_tree.selection_set(items[0].id)
        app._on_tree_select()
        app.update()

        self.assertEqual(len(app.file_tree.get_children()), 100)
        self.assertIn(items[0].error, app.txt_detail.get("1.0", "end-1c"))
        self.assertEqual(heights, {name: getattr(app, name).winfo_height() for name in heights},
                         "Long file details must scroll instead of changing layout heights")
        self.assert_main_controls_visible(app)
        self.assert_two_rows_visible(app)

    def test_expanded_long_list_and_details_at_100_percent(self):
        self.check_long_list(1.0, expanded=True)

    def test_expanded_long_list_and_details_at_150_percent(self):
        self.check_long_list(1.5, expanded=True)

    def test_long_list_and_details_at_100_percent(self):
        self.check_long_list(1.0)

    def test_long_list_and_details_at_150_percent(self):
        self.check_long_list(1.5)

    def check_settings_scroll(self, scale):
        app = self.create_app(scale)
        app.tabs.select(1)
        app.update()
        settings = app.nametowidget(app.tabs.tabs()[1])
        canvas = next(child for child in settings.winfo_children() if isinstance(child, tk.Canvas))
        self.assert_inside_window(app, canvas, "settings canvas")
        self.assertGreater(canvas.bbox("all")[3], canvas.winfo_height())
        canvas.yview_moveto(1)
        app.update()
        self.assertAlmostEqual(canvas.yview()[1], 1.0)
        prompt_top = app.ent_prompt.winfo_rooty() - canvas.winfo_rooty()
        self.assertGreaterEqual(prompt_top, 0)
        self.assertLessEqual(prompt_top + app.ent_prompt.winfo_height(), canvas.winfo_height())
        for name in ("btn_process", "btn_stop", "btn_retry", "lbl_summary", "tabs"):
            self.assert_inside_window(app, getattr(app, name), name)

    def test_compact_settings_scroll_at_100_percent(self):
        self.check_settings_scroll(1.0)

    def test_compact_settings_scroll_at_150_percent(self):
        self.check_settings_scroll(1.5)


if __name__ == "__main__":
    unittest.main()
