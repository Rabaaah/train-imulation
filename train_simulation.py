import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# ========================================
# מחלקת רכבת - תנועה, בלימה, תחנות, רמזורים
# ========================================
class Train:
    """
    מייצגת רכבת אחת במערכת: מהירות, מיקום, בלימה, עצירה בתחנות, תגובה לרמזורים, ותיעוד סטטיסטי.
    """

    def __init__(self, train_id, start_position, speed, max_speed, length=50, color='blue'):
        """
        אתחול רכבת עם מזהה, מיקום התחלתי, מהירות נוכחית, מקסימלית, צבע, אורך ועוד.
        """
        self.train_id = train_id
        self.position = start_position
        self.speed = speed
        self.max_speed = max_speed
        self.length = length
        self.color = color
        self.acceleration = 0.5  # תאוצה (מ'/ש²)
        self.deceleration = -1.0  # בלימה (מ'/ש²)
        self.target_speed = speed
        self.safety_distance = 100  # מרחק בטיחות מינימלי
        self.station_stop_time = 0  # זמן עצירה בתחנה
        self.total_stops = 0
        self.delay = 0
        self.fuel_consumption = 0
        self.history = {'time': [], 'position': [], 'speed': [], 'fuel': []}
        self.rect = None
        self.text = None

    def update_position(self, dt, other_trains, stations, traffic_lights):
        """
        עדכון מיקום הרכבת בכל צעד זמן:
        - בדיקת מרחקי בטיחות מול רכבות אחרות
        - עצירה בתחנות לפי הצורך
        - תגובה לרמזורים
        - עדכון מהירות בהתאם ליעד
        - חישוב צריכת דלק
        """
        self._enforce_safety_distance(other_trains)
        self._handle_station_stop(stations)
        self._handle_traffic_lights(traffic_lights)
        self._update_speed(dt)

        if self.station_stop_time <= 0:
            self.position += self.speed * dt

        self.fuel_consumption += abs(self.speed) * 0.01 * dt

        self.history['time'].append(len(self.history['time']))
        self.history['position'].append(self.position)
        self.history['speed'].append(self.speed)
        self.history['fuel'].append(self.fuel_consumption)

    def _enforce_safety_distance(self, other_trains):
        """
        בדיקה אם רכבת קרובה מדי לרכבת אחרת. אם כן - האטה או עצירה.
        """
        for train in other_trains:
            if train.train_id != self.train_id:
                gap = train.position - self.position
                if 0 < gap < self.safety_distance:
                    self.target_speed = min(train.speed * 0.8, self.target_speed)
                    if gap < self.safety_distance * 0.5:
                        self.target_speed = 0

    def _handle_station_stop(self, stations):
        """
        בדיקה אם צריך לעצור בתחנה קרובה.
        """
        if self.station_stop_time > 0:
            self.station_stop_time -= 1
            return
        for station in stations:
            if abs(self.position - station.position) < 20 and self.speed > 0:
                self.station_stop_time = station.stop_duration
                self.total_stops += 1
                self.target_speed = 0

    def _handle_traffic_lights(self, traffic_lights):
        """
        עצירה אם יש רמזור אדום קרוב.
        """
        for light in traffic_lights:
            if abs(self.position - light.position) < 50 and light.is_red and self.speed > 0:
                self.target_speed = 0

    def _update_speed(self, dt):
        """
        התאמת מהירות הרכבת בהתאם למהירות היעד.
        """
        if self.speed < self.target_speed:
            self.speed = min(self.speed + self.acceleration * dt, min(self.target_speed, self.max_speed))
        elif self.speed > self.target_speed:
            self.speed = max(self.speed + self.deceleration * dt, max(self.target_speed, 0))

# ========================================
# מחלקת תחנה - שם, מיקום, זמן עצירה
# ========================================
class Station:
    """
    מייצגת תחנת רכבת במסילה עם מיקום וזמן עצירה קבוע.
    """
    def __init__(self, name, position, stop_duration=30):
        self.name = name
        self.position = position
        self.stop_duration = stop_duration
        self.rect = None
        self.text = None

# ========================================
# מחלקת רמזור - שליטה על תנועה
# ========================================
class TrafficLight:
    """
    מייצגת רמזור על המסילה – עובר בין אדום לירוק לפי זמנים שהוגדרו מראש.
    """
    def __init__(self, position, red_duration=50, green_duration=100):
        self.position = position
        self.red_duration = red_duration
        self.green_duration = green_duration
        self.current_time = 0
        self.is_red = True
        self.circle = None

    def update(self):
        """
        עדכון מצב הרמזור לפי הזמן – מעבר בין ירוק לאדום.
        """
        self.current_time += 1
        if self.is_red and self.current_time >= self.red_duration:
            self.is_red = False
            self.current_time = 0
        elif not self.is_red and self.current_time >= self.green_duration:
            self.is_red = True
            self.current_time = 0

# ========================================
# המחלקה הראשית - סימולציית רכבות חיה
# ========================================
class LiveRailwaySimulation:
    """
    מערכת הדמיה בזמן אמת למסילת רכבת: ניהול רכבות, תחנות, רמזורים, תנועה וסטטיסטיקות.
    """

    def __init__(self, track_length=2000):
        self.track_length = track_length
        self.trains = []
        self.stations = []
        self.traffic_lights = []
        self.time_step = 0
        self.dt = 1.0
        self.is_running = True

        # יצירת הגרף
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.ax.set_xlim(-50, track_length + 50)
        self.ax.set_ylim(-150, 200)
        self.ax.set_xlabel('Distance (m)', fontsize=12)
        self.ax.set_ylabel('Track', fontsize=12)
        self.ax.set_title('Live Train Simulation', fontsize=16, fontweight='bold')
        self.track_line = self.ax.plot([0, track_length], [0, 0], 'k-', linewidth=4)[0]

        self.info_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        self._setup_controls()

        # סטטיסטיקות
        self.statistics = {
            'total_distance': 0,
            'average_speed': [],
            'delays': [],
            'fuel_consumption': [],
            'safety_incidents': 0,
            'throughput': 0
        }

    def _setup_controls(self):
        """
        יצירת כפתור 'Start/Stop' לשליטה בזמן הריצה.
        """
        from matplotlib.widgets import Button
        ax_button = plt.axes([0.85, 0.01, 0.1, 0.05])
        self.start_stop_button = Button(ax_button, 'Start/Stop')
        self.start_stop_button.on_clicked(self._toggle_simulation)

    def _toggle_simulation(self, event):
        """
        מעבר בין מצב ריצה להפסקה.
        """
        self.is_running = not self.is_running

    def add_train(self, train):
        """
        הוספת רכבת לסימולציה, כולל גרפיקה (מלבן + טקסט).
        """
        self.trains.append(train)
        train.rect = Rectangle((train.position - train.length / 2, -10), train.length, 20,
                               facecolor=train.color, edgecolor='black', linewidth=2, alpha=0.8)
        self.ax.add_patch(train.rect)
        train.text = self.ax.text(train.position, 30, f'Train {train.train_id}',
                                  ha='center', va='bottom', fontsize=9, fontweight='bold')

    def add_station(self, station):
        """
        הוספת תחנה כולל גרפיקה.
        """
        self.stations.append(station)
        station.rect = Rectangle((station.position - 30, -25), 60, 15,
                                 facecolor='lightblue', edgecolor='navy', linewidth=2, alpha=0.7)
        self.ax.add_patch(station.rect)
        station.text = self.ax.text(station.position, -35, station.name,
                                    ha='center', va='top', fontsize=10, fontweight='bold')

    def add_traffic_light(self, light):
        """
        הוספת רמזור למסילה.
        """
        self.traffic_lights.append(light)
        color = 'red' if light.is_red else 'green'
        light.circle = Circle((light.position, 50), 15, facecolor=color, edgecolor='black', linewidth=2)
        self.ax.add_patch(light.circle)

    def update_animation(self, frame):
        """
        הפונקציה שמריצה את הסימולציה בזמן אמת. מתבצעת בכל פריים.
        """
        if not self.is_running:
            return []

        self.time_step += 1

        for light in self.traffic_lights:
            light.update()
            if light.circle:
                light.circle.set_facecolor('red' if light.is_red else 'green')

        for train in self.trains:
            train.target_speed = train.max_speed
            train.update_position(self.dt, self.trains, self.stations, self.traffic_lights)

            if train.rect:
                train.rect.set_x(train.position - train.length / 2)
                speed_ratio = train.speed / train.max_speed if train.max_speed else 0
                if speed_ratio > 0.8:
                    train.rect.set_facecolor('green')
                elif speed_ratio > 0.4:
                    train.rect.set_facecolor('orange')
                else:
                    train.rect.set_facecolor('red')

            if train.text:
                train.text.set_position((train.position, 30))
                train.text.set_text(f'Train {train.train_id}\n{train.speed:.1f} m/s')

        self._update_info_display()
        self._update_statistics()

        if all(train.position >= self.track_length for train in self.trains):
            print("All trains have reached the final station. Ending simulation.")
            plt.close(self.fig)

        return []

    def _update_info_display(self):
        """
        עדכון תיבת המידע הסטטית בצד הגרף.
        """
        lines = [
            f'Time: {self.time_step}',
            f'Trains: {len(self.trains)}',
            f'Stations: {len(self.stations)}',
            f'Traffic Lights: {len(self.traffic_lights)}',
            '', 'Train Status:'
        ]
        for train in self.trains:
            status = 'Stopped' if train.speed < 0.1 else 'Moving'
            lines.append(f'  {train.train_id}: {train.position:.0f} m, {train.speed:.1f} m/s ({status})')

        lines.extend(['', 'Lights:'])
        for i, light in enumerate(self.traffic_lights):
            color = 'Red' if light.is_red else 'Green'
            lines.append(f'  Light {i+1}: {color}')

        self.info_text.set_text('\n'.join(lines))

    def _update_statistics(self):
        """
        עדכון סטטיסטיקות כלליות כמו מהירות ממוצעת וצריכת דלק.
        """
        speeds = [t.speed for t in self.trains if t.speed > 0]
        if speeds:
            self.statistics['average_speed'].append(np.mean(speeds))
        total_fuel = sum(t.fuel_consumption for t in self.trains)
        self.statistics['fuel_consumption'].append(total_fuel)

    def start_animation(self, interval=100):
        """
        התחלת הסימולציה בפועל. תומך בהפעלת אנימציה אינטראקטיבית.
        """
        legend_items = [
            plt.Line2D([0], [0], color='green', lw=4, label='High Speed'),
            plt.Line2D([0], [0], color='orange', lw=4, label='Medium Speed'),
            plt.Line2D([0], [0], color='red', lw=4, label='Low/Stopped Speed'),
            plt.Rectangle((0,0),1,1, facecolor='lightblue', label='Station'),
            plt.Circle((0,0),1, facecolor='red', label='Red Light'),
            plt.Circle((0,0),1, facecolor='green', label='Green Light')
        ]
        self.ax.legend(handles=legend_items, loc='upper right')

        self.anim = animation.FuncAnimation(
            self.fig, self.update_animation,
            interval=interval, blit=False, repeat=True, cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()
        return self.anim

    def show_statistics_dashboard(self):
        """
        הצגת לוח בקרה עם גרפים סטטיסטיים בסוף הסימולציה.
        """
        if not self.statistics['average_speed']:
            print("No statistics data available.")
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Simulation Statistics Dashboard', fontsize=16, fontweight='bold')

        axes[0,0].plot(self.statistics['average_speed'], linewidth=2)
        axes[0,0].set_title('Average Speed Over Time')
        axes[0,1].plot(self.statistics['fuel_consumption'], linewidth=2)
        axes[1,0].hist([t.position for t in self.trains], bins=10, alpha=0.7)
        axes[1,1].axis('off')

        stats_text = f"""
Simulation Steps: {self.time_step}
Total Trains: {len(self.trains)}
Total Stations: {len(self.stations)}
Average Position: {np.mean([t.position for t in self.trains]):.1f} m
"""
        axes[1,1].text(0.1, 0.9, stats_text, transform=axes[1,1].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.tight_layout()
        plt.show()

# ========================================
# יצירת הדגמת סימולציה
# ========================================
def create_demo_simulation():
    """
    יצירת סימולציה מוכנה עם רכבות, תחנות ורמזורים.
    """
    sim = LiveRailwaySimulation(track_length=2000)
    stations = [
        Station("Start Station", 200, 40),
        Station("Middle Station", 600, 30),
        Station("North Station", 1000, 35),
        Station("End Station", 1400, 25)
    ]
    for s in stations:
        sim.add_station(s)

    lights = [
        TrafficLight(400, 60, 120),
        TrafficLight(800, 45, 90),
        TrafficLight(1200, 50, 100)
    ]
    for l in lights:
        sim.add_traffic_light(l)

    colors = ['blue', 'red', 'green', 'purple']
    trains = [
        Train("T01", 0, 15, 30, 50, colors[0]),
        Train("T02", -150, 14, 28, 45, colors[1]),
        Train("T03", -300, 18, 32, 55, colors[2]),
        Train("T04", -450, 13, 25, 40, colors[3])
    ]
    for t in trains:
        sim.add_train(t)

    return sim

# ========================================
# הפעלה בפועל
# ========================================
if __name__ == "__main__":
    print("=== Starting Live Train Simulation Demo ===")
    simulation = create_demo_simulation()
    print("Press the Start/Stop button to pause or resume.")
    anim = simulation.start_animation(interval=100)
    print("Simulation ended. Showing statistics dashboard...")
    simulation.show_statistics_dashboard()


