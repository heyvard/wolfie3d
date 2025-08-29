#!/usr/bin/env python3
"""
Bullet and projectile classes for Wolfie3D game
"""

from ..world.map import is_wall


class Bullet:
    """Standard bullet projectile."""
    
    def __init__(self, x: float, y: float, vx: float, vy: float) -> None:
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.alive = True
        self.age = 0.0
        self.height_param = 0.2  # 0..~0.65 (stiger visuelt)

    def update(self, dt: float) -> None:
        if not self.alive:
            return
        nx = self.x + self.vx * dt
        ny = self.y + self.vy * dt
        if is_wall(int(nx), int(ny)):
            self.alive = False
            return
        self.x, self.y = nx, ny
        self.age += dt
        self.height_param = min(0.65, self.height_param + 0.35 * dt)


class SauBullet:
    """Sau som sendes som prosjektil med bounce og eksplosjon"""
    
    def __init__(self, x: float, y: float, vx: float, vy: float) -> None:
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.alive = True
        self.age = 0.0
        self.height_param = 0.3  # Litt høyere enn kuler
        self.bounce_count = 0  # Antall bounces
        self.max_bounces = 3   # Maksimalt antall bounces
        self.bounce_speed = 0.8  # Fart etter bounce (80% av original)
        self.explosion_ready = True  # Kan eksplodere ved neste space-klikk (alltid klar)
        self.explosion_radius = 2.5  # Eksplosjonsradius

    def update(self, dt: float) -> None:
        if not self.alive:
            return
        nx = self.x + self.vx * dt
        ny = self.y + self.vy * dt
        if is_wall(int(nx), int(ny)):
            self.alive = False
            return
        self.x, self.y = nx, ny
        self.age += dt
        self.height_param = min(0.7, self.height_param + 0.3 * dt)

    def bounce(self, enemy_x: float, enemy_y: float) -> None:
        """Bounce sauen bort fra fienden"""
        if self.bounce_count >= self.max_bounces:
            self.alive = False
            return
            
        # Beregn retning fra fienden til sauen
        dx = self.x - enemy_x
        dy = self.y - enemy_y
        dist = (dx * dx + dy * dy) ** 0.5
        
        if dist > 0:
            # Normaliser retning
            dx /= dist
            dy /= dist
            
            # Sett ny hastighet i bounce-retning
            self.vx = dx * abs(self.vx) * self.bounce_speed
            self.vy = dy * abs(self.vy) * self.bounce_speed
            
            # Flytt sauen litt bort fra fienden for å unngå kollisjon
            self.x = enemy_x + dx * 0.5
            self.y = enemy_y + dy * 0.5
            
            self.bounce_count += 1
            # self.explosion_ready forblir True - sauen kan alltid eksplodere
            print(f"Sau bouncer! Bounces igjen: {self.max_bounces - self.bounce_count}")

    def explode(self, enemies: list) -> list:
        """Eksploderer og dreper fiender i nærheten"""
        killed_enemies = []
        
        for enemy in enemies:
            if not enemy.alive:
                continue
                
            # Beregn avstand til eksplosjonen
            dx = enemy.x - self.x
            dy = enemy.y - self.y
            dist = (dx * dx + dy * dy) ** 0.5
            
            # Hvis fienden er innenfor eksplosjonsradius
            if dist <= self.explosion_radius:
                enemy.hp = 0  # Drept umiddelbart
                enemy.is_dying = True
                enemy.death_time = 0.0
                killed_enemies.append(enemy)
                print(f"Fiende drept av eksplosjon! Avstand: {dist:.2f}")
        
        self.alive = False  # Sauen forbrukes ved eksplosjon
        return killed_enemies
