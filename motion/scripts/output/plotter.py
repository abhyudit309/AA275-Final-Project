import matplotlib.pyplot as plt

def read_file(file_path):
    time, x_values, y_values = [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            t, x, y = map(float, line.split())
            time.append(t)
            x_values.append(x)
            y_values.append(y)
    return time, x_values, y_values

def plot_from_files(file_paths, labels):
    plt.figure(figsize=(8, 6))
    i = 0
    for file_path in file_paths:
        time, x_values, _ = read_file(file_path)
        plt.plot(time, x_values, label=f'{labels[i]}')
        i += 1

    plt.xlabel('time (sec)')
    plt.ylabel('x (m)')
    plt.title("Plot of robot's x-position versus time")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    i = 0
    for file_path in file_paths:
        time, _, y_values = read_file(file_path)
        plt.plot(time, y_values, label=f'{labels[i]}')
        i += 1

    plt.xlabel('time (sec)')
    plt.ylabel('y (m)')
    plt.title("Plot of robot's y-position versus time")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    file_paths = ['ekf_gt_pose.txt', 'ekf_local_pose.txt', 'ekf_ol_pose.txt']
    labels = ['ground truth', 'localization', 'open loop']
    plot_from_files(file_paths, labels)
