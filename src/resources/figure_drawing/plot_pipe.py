import matplotlib.pyplot as plt
import numpy as np

directory = '../../../results/bw_profiling_test/'


exec(open(directory + 'opt-rank0-pci4/statistics/pre_dealloc.py').read())

data1 = gpu2ssd_bw
data2 = ssd2gpu_bw


# Determine the aspect ratio
height = 2  # Keeping height small but large enough to accommodate two pipes
width = 20  # Wider figure to maintain h:w ratio


# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height))

# Plot the first pipe in the first subplot
ax1.imshow([data1], cmap='Blues', aspect='auto')
ax1.axis('off')  # Remove axes

# Plot the second pipe in the second subplot
ax2.imshow([data2], cmap='Blues', aspect='auto')
ax2.axis('off')  # Remove axes


# Set xticks for both subplots
xticks = np.linspace(0, len(data1) - 1, num=10, dtype=int)  # Example: 10 evenly spaced ticks
ax1.set_xticks(xticks)
ax2.set_xticks(xticks)

# Optionally, label the xticks
ax1.set_xticklabels([str(i) for i in xticks])
ax2.set_xticklabels([str(i) for i in xticks])


# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure to a file
plt.savefig('two_pipes_gpu0.png', bbox_inches='tight', pad_inches=0, dpi=300)

# Optionally, show the plot
# plt.show()










exec(open(directory + 'opt-rank2-pci4/statistics/pre_dealloc.py').read())

data1 = gpu2ssd_bw
data2 = ssd2gpu_bw


# Determine the aspect ratio
height = 2  # Keeping height small but large enough to accommodate two pipes
width = 20  # Wider figure to maintain h:w ratio


# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height))

# Plot the first pipe in the first subplot
ax1.imshow([data1], cmap='Blues', aspect='auto')
ax1.axis('off')  # Remove axes

# Plot the second pipe in the second subplot
ax2.imshow([data2], cmap='Blues', aspect='auto')
ax2.axis('off')  # Remove axes


# Set xticks for both subplots
xticks = np.linspace(0, len(data1) - 1, num=10, dtype=int)  # Example: 10 evenly spaced ticks
ax1.set_xticks(xticks)
ax2.set_xticks(xticks)

# Optionally, label the xticks
ax1.set_xticklabels([str(i) for i in xticks])
ax2.set_xticklabels([str(i) for i in xticks])


# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure to a file
plt.savefig('two_pipes_gpu2.png', bbox_inches='tight', pad_inches=0, dpi=300)

# Optionally, show the plot
# plt.show()






exec(open(directory + 'opt-rank4-pci4/statistics/pre_dealloc.py').read())

data1 = gpu2ssd_bw
data2 = ssd2gpu_bw


# Determine the aspect ratio
height = 2  # Keeping height small but large enough to accommodate two pipes
width = 20  # Wider figure to maintain h:w ratio


# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height))

# Plot the first pipe in the first subplot
ax1.imshow([data1], cmap='Blues', aspect='auto')
ax1.axis('off')  # Remove axes

# Plot the second pipe in the second subplot
ax2.imshow([data2], cmap='Blues', aspect='auto')
ax2.axis('off')  # Remove axes

# Set xticks for both subplots
xticks = np.linspace(0, len(data1) - 1, num=10, dtype=int)  # Example: 10 evenly spaced ticks
ax1.set_xticks(xticks)
ax2.set_xticks(xticks)

# Optionally, label the xticks
ax1.set_xticklabels([str(i) for i in xticks])
ax2.set_xticklabels([str(i) for i in xticks])

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure to a file
plt.savefig('two_pipes_gpu4.png', bbox_inches='tight', pad_inches=0, dpi=300)

# Optionally, show the plot
# plt.show()








exec(open(directory + 'opt-rank6-pci4/statistics/pre_dealloc.py').read())

data1 = gpu2ssd_bw
data2 = ssd2gpu_bw


# Determine the aspect ratio
height = 2  # Keeping height small but large enough to accommodate two pipes
width = 20  # Wider figure to maintain h:w ratio


# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height))

# Plot the first pipe in the first subplot
ax1.imshow([data1], cmap='Blues', aspect='auto')
ax1.axis('off')  # Remove axes

# Plot the second pipe in the second subplot
ax2.imshow([data2], cmap='Blues', aspect='auto')
ax2.axis('off')  # Remove axes

# Set xticks for both subplots
xticks = np.linspace(0, len(data1) - 1, num=10, dtype=int)  # Example: 10 evenly spaced ticks
ax1.set_xticks(xticks)
ax2.set_xticks(xticks)

# Optionally, label the xticks
ax1.set_xticklabels([str(i) for i in xticks])
ax2.set_xticklabels([str(i) for i in xticks])

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure to a file
plt.savefig('two_pipes_gpu6.png', bbox_inches='tight', pad_inches=0, dpi=300)

# Optionally, show the plot
# plt.show()