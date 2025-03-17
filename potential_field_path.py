import numpy as np
import pygame
import sys


k_a = 1.0
k_r = 100.0
step_size = 0.1
min_distance = 0.1


start = np.array([1.0, 1.0])
goal = np.array([8.0, 6.0])


obstacles = [
    {"rect": [3, 2, 4, 4], "velocity": np.array([0.02, 0.01])},
    {"rect": [5, 5, 7, 6], "velocity": np.array([-0.015, 0.02])},
    {"rect": [2, 6, 5, 7], "velocity": np.array([0.01, -0.02])}
]


def attractive_force(current, goal_p):
    return k_a * (goal_p - current)


def repulsive_force(current, obstacle):
    force = np.array([0.0, 0.0])
    for obs in obstacle:
        x_min, y_min, x_max, y_max = obs["rect"]

        closest_x = max(x_min, min(current[0], x_max))
        closest_y = max(y_min, min(current[1], y_max))
        closest_point = np.array([closest_x, closest_y])

        distance = np.linalg.norm(current - closest_point)

        if distance < 1e-5:
            distance = 1e-5

        if distance < min_distance:
            force += k_r * (1.0 / distance ** 2) * (current - closest_point) / distance
    return force


def update_obstacles(obstacle, world_width, world_height):
    for obs in obstacle:
        random_change = np.random.uniform(-0.005, 0.005, size=2)
        obs["velocity"] += random_change

        max_speed = 0.05
        speed = np.linalg.norm(obs["velocity"])
        if speed > max_speed:
            obs["velocity"] = (obs["velocity"] / speed) * max_speed

        x_min, y_min, x_max, y_max = obs["rect"]
        vx, vy = obs["velocity"]

        # Обновление координат
        new_x_min = x_min + vx
        new_y_min = y_min + vy
        new_x_max = x_max + vx
        new_y_max = y_max + vy

        if new_x_min < 0 or new_x_max > world_width:
            obs["velocity"][0] = -obs["velocity"][0]
            vx = obs["velocity"][0]
            new_x_min = x_min + vx
            new_x_max = x_max + vx

        if new_y_min < 0 or new_y_max > world_height:
            obs["velocity"][1] = -obs["velocity"][1]
            vy = obs["velocity"][1]
            new_y_min = y_min + vy
            new_y_max = y_max + vy

        obs["rect"] = [new_x_min, new_y_min, new_x_max, new_y_max]


def draw_obstacles(screen, obstacle, scale):
    for obs in obstacle:
        x_min, y_min, x_max, y_max = obs["rect"]
        pygame.draw.rect(screen, (0, 0, 0),
                         (x_min * scale, y_min * scale, (x_max - x_min) * scale, (y_max - y_min) * scale))


def draw_path(screen, path, scale):
    if len(path) > 1:
        points = [(int(p[0] * scale), int(p[1] * scale)) for p in path]
        pygame.draw.lines(screen, (0, 0, 255), False, points, 2)


def main():
    pygame.init()

    width, height = 800, 600
    scale = 70
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Планирование пути с динамическими (случайными) препятствиями")

    clock = pygame.time.Clock()
    robot_pos = np.array(start)
    path = [robot_pos.copy()]

    world_width = width / scale
    world_height = height / scale

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        update_obstacles(obstacles, world_width, world_height)
        f = attractive_force(robot_pos, goal) + repulsive_force(robot_pos, obstacles)

        if np.linalg.norm(f) > 0:
            direction = f / np.linalg.norm(f)
        else:
            direction = np.array([0.0, 0.0])
        robot_pos = robot_pos + step_size * direction
        path.append(robot_pos.copy())

        if np.linalg.norm(robot_pos - goal) < min_distance:
            running = False

        screen.fill((200, 200, 200))
        draw_obstacles(screen, obstacles, scale)
        draw_path(screen, path, scale)

        pygame.draw.circle(screen, (255, 0, 0), (int(start[0] * scale), int(start[1] * scale)), 5)
        pygame.draw.circle(screen, (255, 0, 0), (int(goal[0] * scale), int(goal[1] * scale)), 5)
        pygame.draw.circle(screen, (0, 0, 0), (int(robot_pos[0] * scale), int(robot_pos[1] * scale)), 7)

        pygame.display.flip()
        clock.tick(20)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()