import os
import numpy as np
import matplotlib.pyplot as plt

def process_data_wrapper(input_folder, output_folder, final_depth_cm):
    """
    Processes all files in the input folder using `generateNewData` 
    and saves the processed data as CSV files in the output folder.

    Args:
        input_folder (str): Path to the input folder containing files to process.
        output_folder (str): Path to the output folder to save processed files.
        final_depth_cm (float): The final depth in cm (negative value for below surface).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        if os.path.isfile(input_path) and filename.endswith('.txt'):
            try:
                # Process the data
                processed_accel_data, processed_pressure_data = process_data(input_path, final_depth_cm)

                # Save the processed data to a new CSV file in the output folder
                accel_output_path = os.path.join(output_folder, f"processed_accel_{filename}")
                np.savetxt(accel_output_path, processed_accel_data, delimiter=",", fmt="%s")
                print(f"Processed and saved: {accel_output_path}")

                pressure_output_path = os.path.join(output_folder, f"processed_pressure_{filename}")
                np.savetxt(pressure_output_path, processed_pressure_data, delimiter=",", fmt="%s")
                print(f"Processed and saved: {pressure_output_path}")

                
            except Exception as e:
                print(f"Error processing {filename}: {e}")


def process_data(filename, final_depth_cm) -> list:
    # Get csv data
    data = np.loadtxt(filename, delimiter=",", dtype=str)

    # Count sensor readings to get array size
    accelerationSamples = 0
    pressureSamples = 0
    for row in data:
        if (row[0] == "a"):
            accelerationSamples += 1 
        elif(row[0] == "f"):
            pressureSamples += 1

    # Create numpy arrays
    accelTime = np.zeros(shape=(accelerationSamples,1))
    accelVal = np.zeros(shape=(accelerationSamples,1))
    pressureTime = np.zeros(shape=(pressureSamples,1))
    pressureVal = np.zeros(shape=(pressureSamples,1))

    # Find reference zero time
    timeRef = float(data[0][2])

    # Extract and separate data based on sensor
    accelerationSamples = 0
    pressureSamples = 0
    for row in range(len(data)):
        if (data[row][0] == "a"):
            # Convert time from ms to seconds
            accelTime[accelerationSamples] = ((float(data[row][2]) - timeRef) / 1000)
            accelVal[accelerationSamples] = (float(data[row][1]))
            # Check for large sudden deceleration
            if accelerationSamples > 0 and (accelVal[accelerationSamples] - accelVal[accelerationSamples - 1]) < -10:  # Threshold for sudden deceleration
                accelVal[accelerationSamples] = 0
            accelerationSamples += 1
        elif (data[row][0] == "f"):
            pressureTime[pressureSamples] = ((float(data[row][2]) - timeRef) / 1000)
            pressureVal[pressureSamples] = max((float(data[row][1])) * 9.81 / (2.827433 * 10**-5) * -1, 0)  # Multiply by -1 and set negative values to 0
            pressureSamples += 1

    # Integrate twice over acceleration data to get position
    velocity = cumulative_trapezoid(accelVal, accelTime)
    position = cumulative_trapezoid(velocity, accelTime)

    # Adjust position to use final depth as reference
    final_depth_m = final_depth_cm / 100.0
    position_adjustment = final_depth_m - position[-1]
    position += position_adjustment

    # Create time vs position lookup table
    timePositionLookup = np.hstack((accelTime, position))

    pressurePosition = np.zeros(shape=(len(pressureTime),1))
    for row in range(len(pressureTime)):
        pressurePosition[row] = position_lookup(pressureTime[row], timePosLookup=timePositionLookup)

    # Create output arrays
    output_accel_data = np.hstack((timePositionLookup, accelVal))
    output_pressure_data = np.hstack((pressureTime, pressurePosition, pressureVal))

    return output_accel_data, output_pressure_data


def cumulative_trapezoid(y, x) -> np.array:
    """
        Returns the trapezoidal numerical integral values for each point in x
    """
    integral = np.zeros(shape=(len(x),1))
    for i in range(1, len(x)):
        integral[i] = integral[i-1] + (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2

    return integral


def position_lookup(timeVal, timePosLookup):
    """
        Convert a time value to a position based on a lookup table
        Returns interpolated position between closest table values
    """
    if (timeVal <= timePosLookup[0][0]):
        return timePosLookup[0][1]
    elif (timeVal >= timePosLookup[len(timePosLookup)-1][0]):
        return timePosLookup[len(timePosLookup)-1][1]
    
    rowCount = 0
    while timeVal > timePosLookup[rowCount][0]:
        rowCount += 1
    
    interpolation = (timeVal - timePosLookup[rowCount-1][0]) / (timePosLookup[rowCount][0] - timePosLookup[rowCount-1][0])
    position = (timePosLookup[rowCount-1][1] + interpolation * (timePosLookup[rowCount][1] - timePosLookup[rowCount-1][1]))

    return position


# Example usage
final_depth_cm = float(input("What is the final depth of the snow sample? Please answer in cm: "))  # User-provided final depth in cm
process_data_wrapper("raw_test_data", "processed_test_data", final_depth_cm)
output_accel_data, output_pressure_data = process_data("raw_test_data/test3.txt", final_depth_cm)
print(output_accel_data)

# Plot pressureVal vs. position
pressureVal = output_pressure_data[:, 2]
position = output_pressure_data[:, 1]

# Calculate the scaling factor based on non-zero position values
non_zero_positions = position[position != 0]
calculated_final_position = non_zero_positions[-1] if len(non_zero_positions) > 0 else 1
scaling_factor = final_depth_cm / 100.0 / calculated_final_position

# Apply the scaling factor to the position values
scaled_position = position * scaling_factor

plt.plot(pressureVal, scaled_position)
plt.xlabel('Pressure Value')
plt.ylabel('Position')
plt.title(f'Position vs. Pressure Value (Final Depth: {final_depth_cm} cm)')
plt.ylim(-final_depth_cm / 100.0, 0)

# Add vertical lines
plt.axvline(x=1, color='r', linestyle='--', label='Fist')
plt.axvline(x=10**3, color='g', linestyle='--', label='1F')
plt.axvline(x=10**4, color='b', linestyle='--', label='4F')
plt.axvline(x=10**5, color='y', linestyle='--', label='Pencil')
plt.axvline(x=10**6, color='m', linestyle='--', label='Knife')

# Set x-axis to logarithmic scale
plt.xscale('log')

# Set x-axis range to start at 10^2.9
plt.xlim(left=10**2.9)

# Add legend
plt.legend()

plt.show()