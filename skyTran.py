#!/usr/bin/env python3
"""skyTran Software Engineer Test"""

# Python Standard Library Modules
import math
import random
import time

# Python Non-Standard Library Modules
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = 10, 10
from matplotlib.animation import FuncAnimation


# Initialization of Global Variables

# Set global x-y grid size
grid_size = 100.0 # size of x-y (square) grid, i.e., 1000.0 by 1000.0

# Set physical parameters of vehicles 1, 2, and 3
vehicle_length = 5.0 # length of vehicles
dist_between_veh_1_and_veh_2 = 10.0 # distance between vehicles 1 and 2
vehicles_1_and_2_speed = 20.0 # velocity of vehicles 1 and 2
vehicle_3_acceleration = 30.0 # vehicle 3 acceleration
vehicle_3_desired_speed = 20.0 # vehicle 3 desired speed

# Set time increment for simulation
delta_t = 0.01 # seconds


def variable_acceleration(t):
    """
    Generates a variable acceleration proportional to time t using the global
    variable vehicle_3_acceleration;  required for the varying
    acceleration/constant jerk version of the simulator.  Scipy quad
    (integrate) requires a function of this type.

    Parameters
    ----------
    t : float
        time value, etc.

    Returns
    -------
    value of acceleration at time t : float
        acceleration value
    """

    return vehicle_3_acceleration * t


def is_there_a_collision(vehicle_1, vehicle_2, vehicle_3):
    """
    Determines whether or not there is a collision between vehicles 1 and 3 or
    vehicles 2 and 3.

    Parameters
    ----------
    vehicle_1 : list of floats [x, y]
        position of (end of) vehicle 1 in grid.

    vehicle_2 : list of floats [x, y]
        position of (end of) vehicle 2 in grid.

    vehicle_3 : list of floats [x, y]
        position of (end of) vehicle 3 in grid.

    Returns
    -------
    True or False : bool
        boolean value of whether or not a collision has occured given input.
    """

    v1_in = False
    v2_in = False
    v3_in = False

    if vehicle_1[0] <= grid_size / 2.0 and \
            vehicle_1[0] + 5.0 >= grid_size / 2.0:
        v1_in = True

    if vehicle_2[0] <= grid_size / 2.0 and \
            vehicle_2[0] + 5.0 >= grid_size / 2.0:
        v2_in = True

    if vehicle_3[0] <= grid_size / 2.0 and \
            vehicle_3[0] + 5.0 / math.sqrt(2.0) >= grid_size / 2.0:
        v3_in = True

    if v1_in and v3_in:
        return True
    elif v2_in and v3_in:
        return True
    else:
        return False


# Part 1
def simulate_merging_constant_acceleration(initial_start_range):
    """
    Runs the vehicle simulation under the condition of constant acceleration,
    and generates a list of all vehicle positions in time.  Simulation ends
    if either a vehicle contacts the end of the grid or if there is a
    collision.

    Parameters
    ----------
    initial_start_range: float
        % of grid range to confine random start positions for all vehicles.

    Returns
    -------
    vehicle_positions_in_time : list
        list of all vehicle positions in time.
    """

    # Initialize start time to 0.0
    t = 0.0

    x1 = random.randint(0, int(initial_start_range * grid_size))
    x2 = random.randint(0, int(initial_start_range * grid_size))

    veh_1 = [x1, int(0.5 * grid_size)]
    veh_2 = [x1 + vehicle_length + dist_between_veh_1_and_veh_2,
                int(0.5 * grid_size)]
    veh_3 = [x2, x2]

    vehicle_positions_in_time = []

    while veh_2[0] + vehicle_length < grid_size and veh_3[0] <= grid_size:

        vehicle_3_speed = vehicle_3_acceleration * t
        if vehicle_3_speed >= vehicle_3_desired_speed:
            vehicle_3_speed = vehicle_3_desired_speed

        veh_1[0] += delta_t * vehicles_1_and_2_speed
        veh_2[0] += delta_t * vehicles_1_and_2_speed
        veh_3[0] += delta_t * vehicle_3_speed
        veh_3[1] += delta_t * vehicle_3_speed

        tmp_1 = veh_1.copy()
        tmp_2 = veh_2.copy()
        tmp_3 = veh_3.copy()

        vehicle_positions_in_time.append([t, tmp_1, tmp_2, tmp_3])

        if is_there_a_collision(veh_1, veh_2, veh_3):
            return vehicle_positions_in_time

        t += delta_t

    return vehicle_positions_in_time


# Part 2
def test_harness_constant_acceleration(number_of_test_cases):
    """
    Test harness for running constant acceleration version of simulator.

    Parameters
    ----------
    number_of_test_cases : int
        number of simulations to execute.

    Returns
    -------
    None
    """

    start_range = 0.25
    for i in range(number_of_test_cases):
        vehicle_data = simulate_merging_constant_acceleration(start_range)
        veh_1 = vehicle_data[-1][1]
        veh_2 = vehicle_data[-1][2]
        veh_3 = vehicle_data[-1][3]
        if is_there_a_collision(veh_1, veh_2, veh_3):
            print('collision')
            print(veh_1, veh_2, veh_3)

    return None


# Part 3
def simulate_merging_variable_acceleration(x1=0, x2=0):
    """
    Runs the vehicle simulation under the condition of variable acceleration,
    and generates a list of all vehicle positions in time.  Simulation ends
    if either a vehicle contacts the end of the grid or if there is a
    collision.

    Parameters
    ----------
    initial_start_range: float
        % of grid range to confine random start positions for all vehicles.

    Returns
    -------
    vehicle_positions_in_time : list
        list of all vehicle positions in time.
    """

    # Initialize start time to 0.0
    t = 0.0

    veh_1 = [x1, int(0.5 * grid_size)]
    veh_2 = [x1 + vehicle_length + dist_between_veh_1_and_veh_2,
                int(0.5 * grid_size)]
    veh_3 = [x2, x2]

    vehicle_positions_in_time = []

    while veh_2[0] + vehicle_length < grid_size and veh_3[0] <= grid_size:

        var = integrate.quad(variable_acceleration, 0, t)
        vehicle_3_speed = var[0]
        if vehicle_3_speed >= vehicle_3_desired_speed:
            vehicle_3_speed = vehicle_3_desired_speed

        veh_1[0] += delta_t * vehicles_1_and_2_speed
        veh_2[0] += delta_t * vehicles_1_and_2_speed
        veh_3[0] += delta_t * vehicle_3_speed
        veh_3[1] += delta_t * vehicle_3_speed

        tmp_1 = veh_1.copy()
        tmp_2 = veh_2.copy()
        tmp_3 = veh_3.copy()

        vehicle_positions_in_time.append([t, tmp_1, tmp_2, tmp_3])

        if is_there_a_collision(veh_1, veh_2, veh_3):
            return vehicle_positions_in_time

        t += delta_t

    return vehicle_positions_in_time


# Part 4
def test_harness_variable_acceleration(number_of_test_cases):
    """
    Test harness for running variable acceleration version of simulator.

    Parameters
    ----------
    number_of_test_cases : int
        number of simulations to execute.

    Returns
    -------
    None
    """

    start_range = 0.49
    for i in range(number_of_test_cases):
        vehicle_data = simulate_merging_variable_acceleration(start_range)
        veh_1 = vehicle_data[-1][1]
        veh_2 = vehicle_data[-1][2]
        veh_3 = vehicle_data[-1][3]
        if is_there_a_collision(veh_1, veh_2, veh_3):
            print('collision')
            print(veh_1, veh_2, veh_3)

    return None


# Part 5
def merge_animation(animation_title, vehicle_data):
    """
    Function using Matplotlib for plotting simulations; the Matplotlib
    function for animation is used.  Note that the author of this function
    fully admits to using other people's code from Stackoverflow.  This
    function is hacked together to get something working for the purpose of
    the test for skyTran.
    """

    def veh_1(i):
        return np.array([vehicle_data[i][1][0], vehicle_data[i][1][1]])

    def veh_2(i):
        return np.array([vehicle_data[i][2][0], vehicle_data[i][2][1]])

    def veh_3(i):
        return np.array([vehicle_data[i][3][0], vehicle_data[i][3][1]])

    # create a figure with an axes
    fig, ax = plt.subplots()
    plt.suptitle(animation_title, y=1)
    # set the axes limits
    ax.axis([0.0, grid_size, 0.0, grid_size])
    M = ax.transData.get_matrix()
    xscale = M[0,0]
    yscale = M[1,1]

    plt.plot([0.0, grid_size], [0.0, grid_size], 'y-', lw=22)
    plt.plot([0.0, grid_size], [int(grid_size / 2.0), int(grid_size / 2.0)],
            'y-', lw=22)

    # set equal aspect such that the circle is not shown as ellipse
    ax.set_aspect("equal")
    # create a point in the axes
    point_1, = ax.plot(0, 50, marker=(4, 0, 45), markersize=xscale*3.5,
                       label='Vehicle 1')
    point_2, = ax.plot(0, 50, marker=(4, 0, 45), markersize=xscale*3.5,
                       label='Vehicle 2')
    point_3, = ax.plot(0, 0, marker=(4, 0, 0), markersize=xscale*3.5,
                       label='Vehicle 3')

    plt.legend(loc="upper left")

    title = ax.text(0.5*grid_size, 0.85*grid_size, "",
                    bbox={'facecolor':'y', 'alpha':0.5, 'pad':5},
                    transform=ax.transAxes, ha="center")

    # Updating function, to be repeatedly called by the animation
    def update(i):
        # obtain point coordinates
        x1, y1 = veh_1(i)
        x2, y2 = veh_2(i)
        x3, y3 = veh_3(i)

        # set point's coordinates
        point_1.set_data([x1],[y1])
        point_2.set_data([x2],[y2])
        point_3.set_data([x3],[y3])
        return point_1, point_2, point_3,

    # create animation with 1s interval, which is repeated,
    # provide the full circle (0,2pi) as parameters
    ani1 = FuncAnimation(fig, update, interval=100, blit=True, repeat=False,
                        frames=list(range(len(vehicle_data))))

    plt.show()

if __name__ == '__main__':

    print('Vehicle 3 through intersection before both 1 and 2...')
    data = simulate_merging_variable_acceleration(2, 40)
    merge_animation('3 THROUGH INTERSECTION BEFORE BOTH 1 AND 2 [VARIABLE]',
                    data)

    print('Vehicle 3 collison in intersection with 2...')
    data = simulate_merging_variable_acceleration(2, 30)
    merge_animation('3 COLLIDES WITH 2 IN INTERSECTION [VARIABLE]', data)

    print('Vehicle 3 through space between 1 and 2...')
    data = simulate_merging_variable_acceleration(2, 25)
    merge_animation('3 THROUGH SPACE BETWEEN 1 AND 2 [VARIABLE]', data)

    print('Vehicle 3 collison in intersection with 1...')
    data = simulate_merging_variable_acceleration(10, 25)
    merge_animation('3 COLLIDES WITH 1 IN INTERSECTION [VARIABLE]', data)

    print('Vehicle 3 through intersection after both 1 and 2...')
    data = simulate_merging_variable_acceleration(10, 15)
    merge_animation('3 THROUGH INTERSECTION BEFORE AFTER 1 AND 2 [VARIABLE]',
                    data)
