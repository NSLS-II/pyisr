#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "trame>=3.10",
#     "trame-vtklocal",
#     "trame-vuetify",
#     "vtk==9.5.0rc2",
# ]

import sys
import argparse

from trame.app import TrameApp
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vtklocal, vuetify3 as v3

import vtk
from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
from vtkmodules.vtkFiltersCore import vtkRectilinearGridToImageFilter


class VTRViewer(TrameApp):
    def __init__(self, vtr_path: str, server=None):
        super().__init__(server)
        self._vtr_path = vtr_path
        self._setup_vtk()
        self._build_ui()

    def _setup_vtk(self):
        # Read .vtr
        reader = vtkXMLRectilinearGridReader()
        reader.SetFileName(self._vtr_path)
        reader.Update()
        rgrid = reader.GetOutput()

        # Convert to image data for volume rendering
        converter = vtkRectilinearGridToImageFilter()
        converter.SetInputData(rgrid)
        converter.Update()
        self.image = converter.GetOutput()

        # VTK volume pipeline
        self.ren = vtk.vtkRenderer()
        self.rw = vtk.vtkRenderWindow()
        self.rw.AddRenderer(self.ren)
        self.rw.OffScreenRenderingOn()

        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(self.image)

        prop = vtk.vtkVolumeProperty()
        prop.SetInterpolationTypeToLinear()
        prop.ShadeOff()

        # simple linear opacity from min→max
        otf = vtk.vtkPiecewiseFunction()
        otf.AddPoint(0.0, 0.0)
        otf.AddPoint(1.0, 1.0)
        prop.SetScalarOpacity(otf)

        vol = vtk.vtkVolume()
        vol.SetMapper(mapper)
        vol.SetProperty(prop)

        self.ren.AddVolume(vol)
        self.ren.SetBackground(0.1, 0.1, 0.1)
        self.ren.ResetCamera()

    def _build_ui(self):
        s = self.state
        s.title = f"VTR Viewer: {self._vtr_path}"
        with SinglePageWithDrawerLayout(self.server, title=s.title) as layout:
            with layout.content:
                # server-side VTK rendering
                with vtklocal.LocalView(self.rw) as view:
                    self.ctrl.view_update = view.update
                    self.ctrl.view_reset_camera = view.reset_camera


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trame viewer for .vtr rectilinear grid volume."
    )
    parser.add_argument(
        "vtr_file",
        help="Path to .vtr file (vtkXMLRectilinearGrid format).",
    )
    parser.add_argument(
        "--server", "-s", action="store_true",
        help="Run in server mode without auto-opening the browser.",
    )
    args = parser.parse_args()

    # launch trame app
    VTRViewer(args.vtr_file, server=not args.server).start()