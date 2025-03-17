import math
import random

import numpy as np
import pygame
import skfuzzy as fuzz

angle_range = np.arange(-180, 181, 1)
steering_range = np.arange(-30, 31, 1)

angle_NL = fuzz.trimf(angle_range, [-180, -180, -90])
angle_NS = fuzz.trimf(angle_range, [-180, -90, 0])
angle_Z = fuzz.trimf(angle_range, [-90, 0, 90])
angle_PS = fuzz.trimf(angle_range, [0, 90, 180])
angle_PL = fuzz.trimf(angle_range, [90, 180, 180])

steer_NL = fuzz.trimf(steering_range, [-30, -30, -15])
steer_NS = fuzz.trimf(steering_range, [-30, -15, 0])
steer_Z = fuzz.trimf(steering_range, [-15, 0, 15])
steer_PS = fuzz.trimf(steering_range, [0, 15, 30])
steer_PL = fuzz.trimf(steering_range, [15, 30, 30])


def fuzzy_controller(angle_error):
    """
    Нечеткий контроллер: на основе входной ошибки направления (angle_error)
    вычисляется корректировка угла (steering) с использованием нечеткой логики.
    """
    degree_NL = fuzz.interp_membership(angle_range, angle_NL, angle_error)
    degree_NS = fuzz.interp_membership(angle_range, angle_NS, angle_error)
    degree_Z = fuzz.interp_membership(angle_range, angle_Z, angle_error)
    degree_PS = fuzz.interp_membership(angle_range, angle_PS, angle_error)
    degree_PL = fuzz.interp_membership(angle_range, angle_PL, angle_error)

    out_NL = np.fmin(degree_NL, steer_NL)
    out_NS = np.fmin(degree_NS, steer_NS)
    out_Z = np.fmin(degree_Z, steer_Z)
    out_PS = np.fmin(degree_PS, steer_PS)
    out_PL = np.fmin(degree_PL, steer_PL)

    aggregated = np.fmax(out_NL,
                         np.fmax(out_NS,
                                 np.fmax(out_Z,
                                         np.fmax(out_PS, out_PL))))

    steering = fuzz.defuzz(steering_range, aggregated, 'centroid')
    return steering


def angle_diff(target_angle, current_angle):
    """
    Вычисляет разницу углов так, чтобы результат был в диапазоне [-180, 180].
    """
    diff = target_angle - current_angle
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


class DynamicObstacle:
    """
    Класс динамического препятствия: движущаяся окружность, отскакивающая от границ окна.
    """

    def __init__(self, pos, vel, radius=20):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.radius = radius

    def update(self, dt, width, height):
        self.pos += self.vel * dt

        if self.pos[0] - self.radius < 0 or self.pos[0] + self.radius > width:
            self.vel[0] *= -1

        if self.pos[1] - self.radius < 0 or self.pos[1] + self.radius > height:
            self.vel[1] *= -1

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 0, 255), self.pos.astype(int), self.radius)


def main():
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Fuzzy Logic System Path")
    clock = pygame.time.Clock()

    pos, angle, speed = np.array([100.0, 100.0]), 0.0, 70.0
    target = np.array([np.random.randint(width // 2, width), np.random.randint(height // 2, height)])
    target_threshold = 10.0

    path = [tuple(pos)]

    num_obstacles = 15
    obstacles = []
    for _ in range(num_obstacles):
        obs_pos = [random.randint(50, width - 50), random.randint(50, height - 50)]
        obs_vel = [random.uniform(-50, 50), random.uniform(-50, 50)]
        obstacles.append(DynamicObstacle(obs_pos, obs_vel, radius=20))

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for obs in obstacles:
            obs.update(dt, width, height)

        if np.linalg.norm(target - pos) < target_threshold:
            running = False

        delta = target - pos
        target_angle = math.degrees(math.atan2(delta[1], delta[0]))

        error = angle_diff(target_angle, angle)
        target_steering = fuzzy_controller(error)

        avoidance_sum = 0.0
        detection_radius = 100.0

        for obs in obstacles:
            obs_vector = obs.pos - pos
            d = np.linalg.norm(obs_vector) - obs.radius

            if detection_radius > d > 0:
                obs_angle = math.degrees(math.atan2(obs_vector[1], obs_vector[0]))
                rel_angle = angle_diff(obs_angle, angle)
                avoidance_cmd = fuzzy_controller(-rel_angle)
                weight = (detection_radius - d) / detection_radius
                avoidance_sum += avoidance_cmd * weight

        total_steering = target_steering + avoidance_sum
        total_steering = np.clip(total_steering, -30, 30)

        angle += total_steering

        rad = math.radians(angle)
        pos[0] += speed * dt * math.cos(rad)
        pos[1] += speed * dt * math.sin(rad)

        path.append(tuple(pos))

        screen.fill((30, 30, 30))

        pygame.draw.circle(screen, (0, 255, 0), target.astype(int), 8)
        pygame.draw.lines(screen, (255, 255, 0), False, path, 2)

        for obs in obstacles:
            obs.draw(screen)

        pygame.draw.circle(screen, (255, 0, 0), pos.astype(int), 10)
        #end_pos = (pos[0] + 20 * math.cos(rad), pos[1] + 20 * math.sin(rad))
        #pygame.draw.line(screen, (255, 255, 255), pos.astype(int), (int(end_pos[0]), int(end_pos[1])), 2)

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
