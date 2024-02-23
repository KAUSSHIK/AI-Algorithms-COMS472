import re

def parse_time(time_str):
    """Converts a time string into seconds."""
    if time_str == "Timed out":
        return None
    if ">" in time_str:  # Assumes ">15 min" means exactly 15 minutes
        return 15 * 60
    parts = re.findall(r'\d+', time_str)
    if 'milliseconds' in time_str:
        return int(parts[-1]) / 1000  # Convert milliseconds to seconds
    minutes = int(parts[0]) if 'minutes' in time_str else 0
    seconds = int(parts[1]) if 'seconds' in time_str else 0
    return 60 * minutes + seconds

def calculate_and_print_averages(algorithm_data):
    """Calculates and prints averages for collected data."""
    for alg, data in algorithm_data.items():
        avg_nodes = data['total_nodes'] / data['counts']
        avg_time = data['total_time'] / data['time_counts'] if data['time_counts'] > 0 else 0
        print(f"{alg}: Average Nodes Generated: {avg_nodes}, Average Time Taken (s): {avg_time}")

def main(file_path):
    segments = ['--P1', '--P2', '--P3']
    current_segment = 0
    algorithm_data = {}

    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line indicates a new segment
            if line.strip() in segments:
                # Calculate and print averages for the current segment before resetting
                if algorithm_data:  # Check if data exists
                    print(f"Averages for segment {segments[current_segment]}:")
                    calculate_and_print_averages(algorithm_data)
                    print("\n")  # Print a newline for better readability
                    algorithm_data = {}  # Reset data for the next segment
                current_segment += 1
                continue

            if "ALGORITHM USED" in line:
                alg = line.split("=>")[-1].strip()
                if alg not in algorithm_data:
                    algorithm_data[alg] = {'total_nodes': 0, 'total_time': 0.0, 'counts': 0, 'time_counts': 0}
            elif "Total nodes generated" in line:
                nodes = int(line.split(":")[-1].strip())
                algorithm_data[alg]['total_nodes'] += nodes
                algorithm_data[alg]['counts'] += 1
            elif "Total time taken" in line:
                time_str = line.split(":")[-1].strip()
                time_seconds = parse_time(time_str)
                if time_seconds is not None:
                    algorithm_data[alg]['total_time'] += time_seconds
                    algorithm_data[alg]['time_counts'] += 1

    # After finishing the file, calculate and print for the last segment
    if algorithm_data:  # Check if data exists for the last segment
        print(f"Averages for segment {segments[current_segment-1]}:")
        calculate_and_print_averages(algorithm_data)

if __name__ == "__main__":
    file_path = "./Part3Output.txt"
    main(file_path)
