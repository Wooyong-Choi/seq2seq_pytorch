import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
import matplotlib.font_manager as fm

def vizAttn(input_sentence, output_words, attn_weight):
    
    # Set up figure with colorbar
    fig = plt.figure(figsize=(10, 10))
#    fig.suptitle(''.join(input_sentence)+' -- '+''.join(output_words), fontproperties=fontprop)
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn_weight.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(tk.MultipleLocator(1))
    ax.yaxis.set_major_locator(tk.MultipleLocator(1))

    plt.tight_layout()
    plt.show()
    #plt.savefig('{}.png'.format(idx))
    
    
def vizAccumAttn(input_sentence, attn_weight, filename=""):
    
    # Set up figure with colorbar
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn_weight.sum(dim=0, keepdim=True).numpy(), cmap='bone')
    
    # Set up axes
    ax.set_xticklabels([''] + input_sentence)
    ax.tick_params(
        axis='x',
        which='major',
        bottom=False
    )
    ax.tick_params(
        axis='y',
        which='major',
        left=False,
        labelleft=False
    )
    
    # Show label at every tick
    ax.xaxis.set_major_locator(tk.MultipleLocator(1))
    
    plt.tight_layout()
    
    if filename != "":
        plt.savefig(filename)
    else:
        plt.show()