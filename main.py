import pygame
import sys
from swarm_sim import config
from swarm_sim.swarm import Swarm
from swarm_sim.renderer import Renderer, EditorRenderer
from swarm_sim.editor import Editor
from swarm_sim.environment import Environment


def handle_events(swarm, renderer, editor_renderer, editor, state):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

        if state["editor_mode"]:
            result = editor.handle_event(event, pygame.display.get_surface())
            if result == "exit_editor":
                state["editor_mode"] = False
                env = editor.environment
                swarm.set_environment(env)
                if env.waypoints:
                    swarm.set_waypoints(env.waypoints)
                else:
                    swarm.set_waypoints([])
            elif result == "toggle_grid":
                editor_renderer.show_grid = not editor_renderer.show_grid
            continue

        if event.type != pygame.KEYDOWN:
            continue

        mods = pygame.key.get_mods()
        key = event.key

        if key == pygame.K_SPACE:
            state["paused"] = not state["paused"]

        elif key == pygame.K_r:
            swarm.respawn(config.NUM_DRONES)

        elif key == pygame.K_UP:
            swarm.respawn(config.NUM_DRONES + 5)

        elif key == pygame.K_DOWN:
            swarm.respawn(max(1, config.NUM_DRONES - 5))

        elif key == pygame.K_EQUALS or key == pygame.K_PLUS:
            config.MAX_SPEED = min(10.0, config.MAX_SPEED + 0.2)

        elif key == pygame.K_MINUS:
            config.MAX_SPEED = max(0.4, config.MAX_SPEED - 0.2)

        elif key == pygame.K_l:
            renderer.show_lines = not renderer.show_lines

        elif key == pygame.K_k:
            swarm.kill_random()

        elif key == pygame.K_e:
            state["editor_mode"] = True

        elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            weight_keys = ["separation", "alignment", "cohesion", "relay"]
            idx = key - pygame.K_1
            delta = -0.1 if (mods & pygame.KMOD_SHIFT) else 0.1
            wk = weight_keys[idx]
            config.WEIGHTS[wk] = max(0.0, round(config.WEIGHTS[wk] + delta, 2))

    return True


def main():
    pygame.init()
    screen = pygame.display.set_mode(config.WORLD_SIZE)
    pygame.display.set_caption("Drone Swarm Simulation")
    clock = pygame.time.Clock()

    env = Environment()
    swarm = Swarm()
    swarm.set_environment(env)

    renderer = Renderer(swarm)
    renderer.init(screen)

    editor = Editor(env)
    editor.init(pygame.font.SysFont("monospace", 14))

    editor_renderer = EditorRenderer(editor)
    editor_renderer.init(screen)

    state = {"paused": False, "editor_mode": False}

    while True:
        running = handle_events(swarm, renderer, editor_renderer, editor, state)
        if not running:
            break

        if state["editor_mode"]:
            editor_renderer.draw()
        else:
            if not state["paused"]:
                swarm.tick()
            fps = clock.get_fps()
            renderer.draw(fps, state["paused"])

        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
