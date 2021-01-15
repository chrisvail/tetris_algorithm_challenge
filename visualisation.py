import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns
from time import sleep

from utils import generate_target as gt
from utils import check_solution as cs
from main import solve as si


def visualise_solution(target, solution_iter, interval=10):
    fig = plt.figure()
    vmax = 20
    cmap = generate_palette()


    target = [[x*vmax for x in row] for row in target]
    width = len(target[0])
    height = len(target)
    print(f"Width:\t{width}\nHeight:\t{height}")
    ax = sns.heatmap(target, linewidth=0.1, linecolor="k", vmin=0, vmax=vmax, square=True, cmap=cmap, cbar=False)
    ax.set_ylim(height, -0.5)
    ax.set_xticklabels("") #["" for _ in range(width)], minor=False)
    ax.set_yticklabels("") #["" for _ in range(height)], minor=False)
    plt.tick_params(axis="both", length=0)
    fig.tight_layout()

    def init():
        plt.clf()
        ax = sns.heatmap(target, linewidth=0.1, linecolor="k", vmin=0, vmax=vmax, square=True, cmap=cmap, cbar=False)
        ax.set_ylim(height, -0.5)
        ax.set_xticklabels("") #["" for _ in range(width)], minor=False)
        ax.set_yticklabels("") #["" for _ in range(height)], minor=False)
        plt.tick_params(axis="both", length=0)
        fig.tight_layout()

    def animate(i):
        plt.clf()
        sol = [x[:] for x in target]
        for i, row in enumerate(next(solution_iter)):
            for j, item in enumerate(row):
                if item == (0, 0) or item == (20, 0):
                    sol[i][j] = item
                else:
                    sol[i][j] = item - 2
    
        ax = sns.heatmap(sol, linewidth=0.1, linecolor="k", vmin=0, vmax=vmax, square=True, cmap=cmap, cbar=False)
        ax.set_ylim(height, -0.5)
        ax.set_xticklabels("") #["" for _ in range(width)], minor=False)
        ax.set_yticklabels("") #["" for _ in range(height)], minor=False)
        plt.tick_params(axis="both", length=0)
        fig.tight_layout()

    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=interval) 

    plt.show()



def visualise_solution_dlx(target, solution_iter, animation_save_path, interval=10, fps=5, save_count=200):
    fig = plt.figure()
    vmax = 20
    cmap = generate_palette()


    target = [[x*vmax for x in row] for row in target]
    width = len(target[0])
    height = len(target)
    print(f"Width:\t{width}\nHeight:\t{height}")
    ax = sns.heatmap(target, linewidth=0.1, linecolor="k", vmin=0, vmax=vmax, square=True, cmap=cmap, cbar=False)
    ax.set_ylim(height, -0.5)
    ax.set_xticklabels("") #["" for _ in range(width)], minor=False)
    ax.set_yticklabels("") #["" for _ in range(height)], minor=False)
    plt.tick_params(axis="both", length=0)
    fig.tight_layout()

    sol = [x[:] for x in target]

    def init():
        plt.clf()
        ax = sns.heatmap(target, linewidth=0.1, linecolor="k", vmin=0, vmax=vmax, square=True, cmap=cmap, cbar=False)
        ax.set_ylim(height, -0.5)
        ax.set_xticklabels("") #["" for _ in range(width)], minor=False)
        ax.set_yticklabels("") #["" for _ in range(height)], minor=False)
        plt.tick_params(axis="both", length=0)
        fig.tight_layout()

    def animate(i):
        print(f"Frame {i}")
        plt.clf()
        sol = [x[:] for x in next(solution_iter)]

        for i, row in enumerate(sol):
            for j, item in enumerate(row):
                if item == 0 or item == 1:
                    sol[i][j] = item*20
                else:
                    sol[i][j] = item - 2
    
        ax = sns.heatmap(sol, linewidth=0.1, linecolor="k", vmin=0, vmax=vmax, square=True, cmap=cmap, cbar=False)
        ax.set_ylim(height, -0.5)
        ax.set_xticklabels("") #["" for _ in range(width)], minor=False)
        ax.set_yticklabels("") #["" for _ in range(height)], minor=False)
        plt.tick_params(axis="both", length=0)
        fig.tight_layout()

    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=interval, save_count=save_count) 
    anim.save(animation_save_path, fps=fps)
    plt.show()



def show_solution(target, solution, lw=0.1):
    
    # Get characteristics
    width = len(solution[0])
    height = len(solution)
    vmax = 20 #80

    cmap = generate_palette()

    # Revalue target
    target = [[x*vmax for x in row] for row in target]

    # Superimpose solution
    for i, row in enumerate(solution):
        for j, item in enumerate(row):
            if item == (0, 0): continue
            
            # Revalue solved pieces
            target[i][j] = item[0] - 2 #(item[0]//4)*15 + 2 + 3*(item[0]%4)


    fig = plt.figure()
    ax = sns.heatmap(target, linewidth=lw, linecolor="k", vmin=0, vmax=vmax, square=True, cmap=cmap, cbar=False)
    ax.set_ylim(height, -0.5)
    ax.set_xlim(width, -0.5)
    ax.set_xticklabels("") 
    ax.set_yticklabels("") 
    plt.tick_params(axis="both", length=0)
    #fig.tight_layout()
    plt.show()


def show_target_and_solution(target, solution, lw=0.1):
    
    # Get characteristics
    width = len(solution[0])
    height = len(solution)
    vmax = 20 #80

    cmap = generate_palette()

    # Revalue target
    target = [[x*vmax for x in row] for row in target]

    fig = plt.subplot(1,2,1)
    ax = sns.heatmap(target, linewidth=lw, linecolor="k", vmin=0, vmax=vmax, square=True, cmap=cmap, cbar=False)
    ax.set_ylim(height, -0.5)
    ax.set_xlim(width, -0.5)
    ax.set_xticklabels("") 
    ax.set_yticklabels("") 
    plt.tick_params(axis="both", length=0)


    # Superimpose solution
    for i, row in enumerate(solution):
        for j, item in enumerate(row):
            if item == (0, 0): continue
            
            # Revalue solved pieces
            target[i][j] = item[0] - 2 #(item[0]//4)*15 + 2 + 3*(item[0]%4)


    fig = plt.subplot(1,2,2)
    ax = sns.heatmap(target, linewidth=lw, linecolor="k", vmin=0, vmax=vmax, square=True, cmap=cmap, cbar=False)
    ax.set_ylim(height, -0.5)
    ax.set_xlim(width, -0.5)
    ax.set_xticklabels("") 
    ax.set_yticklabels("") 
    plt.tick_params(axis="both", length=0)
    #fig.tight_layout()
    plt.show()


def generate_palette():
    # Create colourmap
    colours = [(0x44/256, 0x44/256, 0x44/256)]
    """ colours_raw = [(63, 80, 109), (82, 101, 143), (105, 132, 178), (122, 152, 224), 
                   (101, 184, 209), (65, 166, 204), (42, 142, 173), (23, 114, 142), 
                   (239,182,67), (232,170,31), (209, 144, 15), (193, 135, 17),
                   (237, 138, 72), (229, 117, 32), (198, 92, 20), (170, 77, 14)] """
    
    base_colours = [(0x12, 0x85, 0xce), (0xee, 0xa0, 0x1a), (0xff, 0x30, 0x2c),  (0xff, 0x76, 0x18)]
    colours_raw = []
    for r,g,b in base_colours:
        for i in range(4):
            colours_raw.append((int(r*0.9**i), int(g*0.9**i), int(b*0.9**i)))
    colours.extend(map(lambda x: (x[0]/256, x[1]/256, x[2]/256), colours_raw))
    
    #for red, green, blue in ((1,0,0), (0, 1, 0), (0, 0, 1), (0, 1, 1)):
    #    for saturation in (0.4, 0.55, 0.7, 0.85):
    #            colours.append((red*saturation, green*saturation, blue*saturation))

    colours.append((0xf7/256, 0xf7/256, 0xf6/256))
    cmap = sns.color_palette(palette=colours, n_colors=18)
    return cmap


if __name__ == "__main__":
    nx, ny = (25, 25)
    density = 0.6
    ignore = {1,2,3}
    target, sol = gt(nx, ny, density, ignore)

    """ target = [[1,0,0,0,0,0,1,0,1,1,0,0,1,1,1],
              [1,0,0,0,1,1,1,0,0,1,0,0,1,0,0],
              [1,1,0,0,0,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],

              [0,1,0,0,1,1,1,0,1,1,0,0,1,0,0],
              [0,1,0,0,0,0,1,0,1,0,0,0,1,1,1],
              [1,1,0,0,0,0,0,0,1,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],

              [1,0,0,0,0,1,0,0,0,1,0,0,1,1,1],
              [1,1,0,0,1,1,1,0,1,1,0,0,0,1,0],
              [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],

              [0,1,1,0,1,0,0,0,1,1,0,0,0,1,0],
              [1,1,0,0,1,1,0,0,0,1,1,0,1,1,0],
              [0,0,0,0,0,1,0,0,0,0,0,0,1,0,0]] """


    sol = si(target)
    total_pieces = sum([sum(row) for row in target])
    valid, missing, excess, error_pieces = cs(target, sol, ignore)
    print(f"Accuracy: {(total_pieces - missing - excess)*100/(total_pieces)}")

    target = [x[::-1] for x in target]
    sol = [x[::-1] for x in sol]
    show_solution(target, sol)
