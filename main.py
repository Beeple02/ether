import pygame
import sys
from swarm_sim import config
from swarm_sim.swarm import Swarm
from swarm_sim.renderer import Renderer, EditorRenderer
from swarm_sim.editor import Editor
from swarm_sim.environment import Environment


def _make_renderer(swarm, screen):
    """Instantiate the correct renderer based on config.RENDER_3D."""
    if config.RENDER_3D:
        from swarm_sim.camera import Camera
        from swarm_sim.renderer_3d import Renderer3D
        cam = Camera(screen.get_size())
        r   = Renderer3D(swarm, cam)
    else:
        r = Renderer(swarm)
    r.init(screen)
    return r


def handle_events(swarm, state):
    """Process events; returns False when quit is requested.

    Renderer is stored in state["renderer"] so F3 can hot-swap it.
    """
    screen   = state["screen"]
    renderer = state["renderer"]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

        if state["editor_mode"]:
            result = state["editor"].handle_event(event, screen)
            if result in ("exit_editor", "scenario_loaded"):
                state["editor_mode"] = False
                env = state["editor"].environment
                swarm.set_environment(env)
                swarm.set_waypoints(env.waypoints if env.waypoints else [])
            elif result == "toggle_grid":
                state["ed_renderer"].show_grid = not state["ed_renderer"].show_grid
            continue

        if event.type != pygame.KEYDOWN:
            continue

        mods = pygame.key.get_mods()
        key  = event.key

        if key == pygame.K_SPACE:
            state["paused"] = not state["paused"]
        elif key == pygame.K_r:
            swarm.respawn(config.NUM_DRONES)
        elif key == pygame.K_UP:
            swarm.respawn(config.NUM_DRONES + 5)
        elif key == pygame.K_DOWN:
            swarm.respawn(max(1, config.NUM_DRONES - 5))
        elif key in (pygame.K_EQUALS, pygame.K_PLUS):
            config.MAX_SPEED = min(10.0, round(config.MAX_SPEED + 0.2, 1))
        elif key == pygame.K_MINUS:
            config.MAX_SPEED = max(0.2, round(config.MAX_SPEED - 0.2, 1))
        elif key == pygame.K_l:
            renderer.show_lines = not renderer.show_lines
        elif key == pygame.K_F3:
            # Hot-swap renderer between 2D and 3D (visual only; sim is unaffected)
            config.RENDER_3D = not config.RENDER_3D
            prev_lines  = renderer.show_lines
            prev_debug  = renderer.debug_mode
            prev_help   = renderer.help_mode
            renderer    = _make_renderer(swarm, screen)
            renderer.show_lines = prev_lines
            renderer.debug_mode = prev_debug
            renderer.help_mode  = prev_help
            state["renderer"]   = renderer
        elif key == pygame.K_k:
            swarm.kill_random()
        elif key == pygame.K_e:
            state["editor_mode"] = True
        elif key == pygame.K_F11:
            pygame.display.toggle_fullscreen()
        elif key == pygame.K_h:
            renderer.help_mode = not renderer.help_mode
        elif key == pygame.K_TAB:
            renderer.debug_mode = not renderer.debug_mode
        elif key == pygame.K_m:
            swarm.advance_phase()
        elif key == pygame.K_v:
            config.CONVERGENCE_ENABLED = not config.CONVERGENCE_ENABLED
        elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            wkeys = ["separation", "alignment", "cohesion", "relay"]
            wk    = wkeys[key - pygame.K_1]
            delta = -0.1 if (mods & pygame.KMOD_SHIFT) else 0.1
            config.WEIGHTS[wk] = max(0.0, round(config.WEIGHTS[wk] + delta, 2))

    return True


def main():
    pygame.init()
    # SCALED flag: game always renders at WORLD_SIZE, pygame handles
    # stretching to actual window / fullscreen — mouse coords stay in world space
    screen = pygame.display.set_mode(config.WORLD_SIZE, pygame.SCALED | pygame.RESIZABLE | pygame.FULLSCREEN)
    pygame.display.set_caption("Drone Swarm Simulation")
    clock = pygame.time.Clock()

    env   = Environment()
    swarm = Swarm()
    swarm.set_environment(env)

    editor      = Editor(env)
    ed_renderer = EditorRenderer(editor)
    ed_renderer.init(screen)

    state = {
        "paused":      False,
        "editor_mode": False,
        "screen":      screen,
        "renderer":    _make_renderer(swarm, screen),
        "editor":      editor,
        "ed_renderer": ed_renderer,
    }

    while True:
        if not handle_events(swarm, state):
            break

        renderer = state["renderer"]

        if state["editor_mode"]:
            ed_renderer.draw()
        else:
            if not state["paused"]:
                swarm.tick()
            renderer.draw(clock.get_fps(), state["paused"])

        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
