"""Compact two-row, three-column journal process diagrams."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def process_grid(steps, footer):
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(3.5,2.65),facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    xs=[.035,.365,.695];ys=[.565,.125];w=.27;h=.37
    positions=[(xs[0],ys[0]),(xs[1],ys[0]),(xs[2],ys[0]),
               (xs[2],ys[1]),(xs[1],ys[1]),(xs[0],ys[1])]
    for i,((title,body),(x,y)) in enumerate(zip(steps,positions),1):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.007,rounding_size=0.02',
            facecolor='#eaf4f1' if i in (1,6) else '#f1f5f8',edgecolor='#9db5c2',lw=.7))
        ax.text(x+.021,y+h-.042,str(i),fontsize=7.5,weight='bold',color='#34796f',va='center')
        ax.text(x+w/2,y+h-.09,title,fontsize=7,weight='bold',ha='center',va='center',color='#203b4c',linespacing=1.15)
        ax.text(x+w/2,y+.12,body,fontsize=6.3,ha='center',va='center',color='#203b4c',linespacing=1.35)
    for i in range(5):
        x,y=positions[i];nx,ny=positions[i+1]
        if i==2: start=(x+w/2,y-.012);end=(nx+w/2,ny+h+.012)
        elif i<2:start=(x+w+.012,y+h/2);end=(nx-.012,ny+h/2)
        else:start=(x-.012,y+h/2);end=(nx+w+.012,ny+h/2)
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='-|>',lw=.9,color='#507080',shrinkA=0,shrinkB=0))
    ax.text(.5,.048,footer,ha='center',va='center',fontsize=6.5,color='#365364',linespacing=1.3)
    return fig

def calibration_grid():
    return process_grid([
        ('Prepare\nreferences','Visuo-tactile images\nLock aligned poses\nReserve validation'),
        ('Fit\nbackground','Background pixels\nPad RGB and glows\nVignette'),
        ('Fit\nillumination','Directions / edges\nRGB light weights\nBounded fitting'),
        ('Fit image\nfilters','RGB blur; vignette\nContrast; gamma\nTraining objective'),
        ('Noise and\ntexture','Native pixels\nNoise components\nTexture statistics'),
        ('Review and save','Training / validation\nInspect errors\nSave configuration'),
    ],'Follow 1–6. Validation images enter review only.\nGeometry stays fixed during fitting.')
