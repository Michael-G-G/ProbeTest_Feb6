import os
import numpy as np

def process_data_wrapper(input_folder, output_folder):
    """
    Processes all files in the input folder using `generateNewData` 
    and saves the processed data as CSV files in the output folder.

    Args:
        input_folder (str): Path to the input folder containing files to process.
        output_folder (str): Path to the output folder to save processed files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        if os.path.isfile(input_path) and filename.endswith('.txt'):
            try:
                # Process the data
                processed_accel_data, processed_force_data = process_data(input_path)

                # Save the processed data to a new CSV file in the output folder
                accel_output_path = os.path.join(output_folder, f"processed_accel_{filename}")
                np.savetxt(accel_output_path, processed_accel_data, delimiter=",", fmt="%s")
                print(f"Processed and saved: {accel_output_path}")

                force_output_path = os.path.join(output_folder, f"processed_force_{filename}")
                np.savetxt(force_output_path, processed_force_data, delimiter=",", fmt="%s")
                print(f"Processed and saved: {force_output_path}")

                
            except Exception as e:
                print(f"Error processing {filename}: {e}")


def process_data(filename) -> list:
    # Get csv data
    data = np.loadtxt(filename, delimiter=",", dtype=str)

    # Count sensor readings to get array size
    accelerationSamples = 0
    forceSamples = 0
    for row in data:
        if (row[0] == "a"):
            accelerationSamples += 1 
        elif(row[0] == "f"):
            forceSamples += 1

    # Create numpy arrays
    accelTime = np.zeros(shape=(accelerationSamples,1))
    accelVal = np.zeros(shape=(accelerationSamples,1))
    forceTime = np.zeros(shape=(forceSamples,1))
    forceVal = np.zeros(shape=(forceSamples,1))

    # Find reference zero time
    timeRef = float(data[0][2])

    # Extract and separate data based on sensor
    accelerationSamples = 0
    forceSamples = 0
    for row in range(len(data)):
        if (data[row][0] == "a"):
            # Convert time from ms to seconds
            accelTime[accelerationSamples] = ((float(data[row][2]) - timeRef) / 1000)
            accelVal[accelerationSamples] = (float(data[row][1]))
            accelerationSamples += 1
        elif (data[row][0] == "f"):
            forceTime[forceSamples] = ((float(data[row][2]) - timeRef) / 1000)
            forceVal[forceSamples] = (float(data[row][1]))
            forceSamples += 1


    # accelVal = threshold_value(data=accelVal, threshold=0.00)

    # Integrate twice over acceleration data to get position
    velocity = cumulative_trapezoid(x=accelTime, y=accelVal, reset_value=50)


    # velocity = zero_velocity(acceleration_data=accelVal, velocity_data=velocity, threshold=30)
    position = cumulative_trapezoid(x=accelTime, y=velocity)

    # Create time vs position lookup table
    timePositionLookup = np.hstack((accelTime, position))

    forcePosition = np.zeros(shape=(len(forceTime),1))
    for row in range(len(forceTime)):
        forcePosition[row] = position_lookup(forceTime[row], timePosLookup=timePositionLookup)

    # Create output arrays
    output_accel_data = np.hstack((timePositionLookup, accelVal))
    output_force_data = np.hstack((forceTime, forcePosition, forceVal))

    return output_accel_data, output_force_data


def threshold_value(data, threshold):
    for i in range(len(data)):
        if abs(data[i][0]) < threshold:
            data[i][0] = 0
    return data


def cumulative_trapezoid(x,y, reset_value=None) -> np.array:
    """
        Returns the trapezoidal numerical integral values for each point in x
    """
    integral = np.zeros(shape=(len(x),1))
    for i in range(1, len(x)):
        if reset_value != None and y[i][0] > reset_value:
            integral[i] = 0
        else:
            integral[i] = integral[i-1] + (x[i-1] - x[i]) * (y[i-1] + y[i]) / 2
        # integral[i] = integral[i-1] + (x[i-1] - x[i]) * (y[i-1] + y[i]) / 2
    return integral

def get_file(file_path):
    if os.path.exits(file_path):
        return np.loadtxt(file_path, delimiter=",", dtype=str)
    print("Error, no file ", file_path, " found")
    return 


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


# process_data_wrapper(input_folder="raw_test_data", output_folder="processed_test_data")


output_accel_data, output_force_data = process_data("raw_test_data/HalfProbe2-1.txt")

print(output_accel_data)