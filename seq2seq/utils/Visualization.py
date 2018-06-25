import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
import matplotlib.font_manager as fm

def vizAttn(input_sentence, output_words, attn_weight):
    path = '/usr/share/fonts/truetype/MS/malgun.ttf'
    fontprop = fm.FontProperties(fname=path, size='medium')
    
    # Set up figure with colorbar
    fig = plt.figure(figsize=(10, 10), dpi = 80)
#    fig.suptitle(''.join(input_sentence)+' -- '+''.join(output_words), fontproperties=fontprop)
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn_weight.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence + ['<EOS>'], rotation=90, fontproperties=fontprop)
    ax.set_yticklabels([''] + output_words, fontproperties=fontprop)

    # Show label at every tick
    ax.xaxis.set_major_locator(tk.MultipleLocator(1))
    ax.yaxis.set_major_locator(tk.MultipleLocator(1))

    plt.tight_layout()
    plt.show()
    #plt.savefig('{}.png'.format(idx))
    
    
def vizAccumAttn(input_sentence, attn_weight, filename=""):
    path = '/usr/share/fonts/truetype/MS/malgun.ttf'
    fontprop = fm.FontProperties(fname=path, size='large')
    
    # Set up figure with colorbar
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn_weight.sum(dim=0, keepdim=True).numpy(), cmap='bone')
    
    # Set up axes
    ax.set_xticklabels([''] + input_sentence, fontproperties=fontprop)
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