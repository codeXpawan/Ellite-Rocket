import glfw
from OpenGL.GL import *
import math
from Object_Loader import Object_Loader


class Window:
    def __init__(self, width: int, height: int, title: str):
        self._width = width
        self._height = height
        self._title = title

        if not glfw.init():
            raise Exception("glfw can not be initialized!")

        self._window = glfw.create_window(
            self._width, self._height, self._title, None, None)

        if not self._window:
            glfw.terminate()
            raise Exception("glfw window can not be created!")

        # Set resize handler
        glfw.set_window_size_callback(self._window, self._on_resize)
        # Set keyboard input handler
        glfw.set_key_callback(self._window, self._on_key_input)
        glfw.make_context_current(self._window)
        glEnable(GL_DEPTH_TEST)   # Enable depth testing for z-culling
        glEnable(GL_BLEND)        # Enable blending for transparency

        self.scene = {
            "platform": Object_Loader("data/platform.obj", 0, -0.45, 0.2, 0.1),
            "rocket": Object_Loader("data/rocket.obj", 0, -0.3, 0, 0.1),
        }

    def _on_resize(self, _window, width, height) -> None:
        self._width, self._height = width, height
        glViewport(0, 0, self._width, self._height)
        self._update_projection()

    def set_color(self):
        glClearColor(0.4, 0.3, 0.2, 1.0)

    def _on_key_input(self, _window, key, _scancode, action, _mode) -> None:
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(self._window, True)

    def _draw_object(self):
        # glUseProgram(self.scene["platform"].shader)
        self.scene["platform"].draw()

        # glUseProgram(self.scene["rocket"].shader)
        self.scene["rocket"].draw()

    def main_loop(self) -> None:
        while not glfw.window_should_close(self._window):
            glfw.poll_events()
            # Clean the Back buffer and Depth buffer
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.set_color()
            self._draw_object()
            glfw.swap_buffers(self._window)


def main():
    window = Window(800, 600, "Ellite Rocket")
    window.main_loop()
    glfw.terminate()


if __name__ == "__main__":
    main()
