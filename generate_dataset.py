#!/usr/bin/env python3
"""
Generate a realistic synthetic dataset of 500 fitness app users and save to fitness_users.csv.

Columns:
- UserID
- Age (18-60)
- Gender (Male/Female)
- AvgDailySteps
- AvgDailyCalories
- WorkoutFrequencyPerWeek
- PreferredWorkoutType (Cardio/Strength/Flexibility)
- SleepHours

Some missing values are seeded to exercise missing-value handling in the notebook.
"""
import csv
import random
import math

random.seed(42)

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

filename = "fitness_users.csv"
n = 500
workout_types = ["Cardio", "Strength", "Flexibility"]

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["UserID","Age","Gender","AvgDailySteps","AvgDailyCalories","WorkoutFrequencyPerWeek","PreferredWorkoutType","SleepHours"])
    for i in range(1, n+1):
        uid = f"U{i:04d}"
        # Deterministic-ish but varied values
        age = 18 + ((i * 13) % 43)  # 18-60
        gender = "Male" if ((i * 7) % 2 == 0) else "Female"
        # Workout frequency 0..7
        wf = (i * 3) % 8
        # Preferred workout type with some bias by wf
        if wf >= 4:
            pref = random.choices(workout_types, weights=(0.5,0.3,0.2))[0]
        else:
            pref = random.choices(workout_types, weights=(0.4,0.35,0.25))[0]
        # Avg daily steps: base depends on wf and age
        base_steps = 3000 + wf * 1200 + ((i * 131) % 8000)
        age_penalty = int((age - 30) * 30) if age > 30 else 0
        steps = clamp(int(base_steps - age_penalty + random.gauss(0, 800)), 800, 20000)
        # Avg daily calories (total daily burned) as function of steps and workout frequency
        cal_from_steps = steps * 0.04  # rough conversion
        base_cal = 1700 + cal_from_steps + wf * 30 + random.gauss(0, 120)
        calories = int(clamp(base_cal, 1400, 3800))
        # Sleep hours 5.0 - 9.5 with small variation
        sleep = clamp(round(5.0 + ((i * 17) % 45) / 10.0 + random.gauss(0,0.6), 1), 4.0, 9.5)
        # Introduce some missing values intentionally
        if i % 37 == 0:
            calories_field = ""
        else:
            calories_field = str(calories)
        if i % 53 == 0:
            sleep_field = ""
        else:
            sleep_field = str(sleep)
        if i % 41 == 0:
            steps_field = ""
        else:
            steps_field = str(steps)
        writer.writerow([uid, age, gender, steps_field, calories_field, wf, pref, sleep_field])

print(f"Generated {filename} with {n} rows.")
