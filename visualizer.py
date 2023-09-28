import ast
import matplotlib.pyplot as plt
import numpy as np

# Open the log file for reading
with open('output-09-28.log', 'r') as file:
    lines = file.readlines()

parsed_data = []

current_info = {}
current_histogram = []
current_histogram_without_norm = []
current_histogram_absorptions = []
current_histogram_absorptions_without_norm = []

# Loop through each line in the file
for line in lines:
    parts = line.strip().split(' - ')
    if len(parts) < 5:
        continue
    
    date_time_str, _, _, _, info_str = parts[:5]
    info = info_str.strip()
    
    if info.startswith('ABSORBTION_PROB'):
        current_info['ABSORBTION_PROB'] = float(info.split('=')[1].strip())
    elif info.startswith('FRACTAL_LEVELS'):
        current_info['FRACTAL_LEVELS'] = int(info.split('=')[1].strip())
    elif info.startswith('N_PARTICLES'):
        current_info['N_PARTICLES'] = int(info.split('=')[1].strip())
    elif info.startswith('STEP'):
        current_info['STEP'] = float(info.split('=')[1].strip())
    elif info.startswith('N_STEPS'):
        current_info['N_STEPS'] = int(info.split('=')[1].strip())
    elif info.startswith('General histogram:'):
        current_histogram = [ast.literal_eval(info.split(':')[1].strip().replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",","))]
    elif info.startswith('General histogram without norm:'):
        current_histogram_without_norm = [ast.literal_eval(info.split(':')[1].strip().replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",","))]
    elif info.startswith('Histogram (only absorptions):'):
        current_histogram_absorptions = [ast.literal_eval(info.split(':')[1].strip().replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",","))]
    elif info.startswith('Histogram (only absorptions) without norm:'):
        current_histogram_absorptions_without_norm = [ast.literal_eval(info.split(':')[1].strip().replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",","))]
    
    if info == 'Application finish':
        current_info['General histogram'] = current_histogram
        current_info['General histogram without norm'] = current_histogram_without_norm
        current_info['Histogram (only absorptions)'] = current_histogram_absorptions
        current_info['Histogram (only absorptions) without norm'] = current_histogram_absorptions_without_norm
        parsed_data.append(current_info)
        current_info = {}
        current_histogram = []
        current_histogram_without_norm = []
        current_histogram_absorptions = []
        current_histogram_absorptions_without_norm = []

# Display parsed data
img = 0
for data in parsed_data:
    fig, ax = plt.subplots()

    # Plot each row of data as a bar
    for i, row in enumerate(data['Histogram (only absorptions) without norm']):
        x = range(len(row))
        ax.bar(x, row, label=f'Row {i + 1}')

    # Set labels and legend
    ax.set_xlabel('Fractal depth')
    ax.xaxis.set_ticks(np.arange(0, len(row), 1))
    ax.set_ylabel('Number of particles')
    ax.set_yscale("log")
    plt.ylim(top=data['N_PARTICLES']*1.1)
    ax.set_title('Absorption = %1.1f' %data['ABSORBTION_PROB'])
    for i in x:
        plt.text(i, row[i], row[i], ha = 'center')

    # Save the plot
    plt.savefig("Absorption%1.1f_Fractal"  %data['ABSORBTION_PROB'] + "%i.png" %data['FRACTAL_LEVELS'])
    img+=1