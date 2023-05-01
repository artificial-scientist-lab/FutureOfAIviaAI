import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()
colors = sns.color_palette()


allplots=[]
all_names=[]


yichao=[    0.9330, 0.9252, 0.9248, \
            0.9172, 0.9191, 0.9096, \
            0.8960, 0.8987, 0.8935, \
            0.9926, 0.9945, 0.9982, \
            0.9853, 0.9965, 0.9949, \
            0.9793, 0.9893, 0.9990]
allplots.append(yichao)   
#all_names.append('yichao')  
all_names.append('NF+ML (M1)') 


HashBrown=[ 0.9147, 0.9175, 0.9156, \
            0.8953, 0.8977, 0.8949, \
            0.8610, 0.8645, 0.8630, \
            0.9900, 0.9876, 0.9944, \
            0.9786, 0.9861, 0.9867, \
            0.9595, 0.9689, 0.9692]
allplots.append(HashBrown)  
#all_names.append('HashBrown')
all_names.append('NF+ML (M2)')  


Nima=[      0.8979, 0.8980, 0.9010, \
            0.8830, 0.8823, 0.8823, \
            0.8489, 0.8433, 0.8409, \
            0.9496, 0.9687, 0.9481, \
            0.9652, 0.9765, 0.9788, \
            0.9480, 0.9538, 0.9488] 
allplots.append(Nima) 
all_names.append('NF+ML (M3)') 


Bacalhink_prefAttachment=[ 
            0.8838, 0.8862, 0.8836, \
            0.8695, 0.8673, 0.8628, \
            0.8422, 0.8359, 0.8300, \
            0.9754, 0.9649, 0.9789, \
            0.9590, 0.9620, 0.9646, \
            0.9380, 0.9442, 0.9386]
allplots.append(Bacalhink_prefAttachment)  
#all_names.append('Bacalhink_prefAttachment')  
all_names.append('pure NF (M4A)') 


SanatisFinests2=[ 0.90484, 0.9099, 0.9068, \
            0.8660, 0.8659, 0.8020, \
            0.6733, 0.7308, 0.7780, \
            0.9872, 0.9852, 0.9915, \
            0.9469, 0.9486, 0.9633, \
            0.7952, 0.7870, 0.6731] 
allplots.append(SanatisFinests2) 
all_names.append('NF+ML (M5)')   



Bacalhink_CommonNeighbor=[ 
            0.8942, 0.9016, 0.9009, \
            0.8476, 0.8761, 0.8783, \
            0.7677, 0.8266, 0.8345, \
            0.9369, 0.9771, 0.9889, \
            0.9247, 0.9760, 0.9786, \
            0.8658, 0.9520, 0.9526]
allplots.append(Bacalhink_CommonNeighbor) 
#all_names.append('Bacalhink_CommonNeighbor')      
all_names.append('pure NF (M4B)') 

    
Valente=[ 0,0,0, \
          0.8467, 0.8490, 0.8335, \
          0.7897, 0.8023, 0.8004, \
          0,0,0.9819,\
          0.9420, 0.9562, 0.9461,\
          0.8914, 0.9262, 0.9150]
allplots.append(Valente)    
#all_names.append('Valente')   
all_names.append('NF+ML (M5)') 

mkbaseline=[0.8520, 0.8526, 0.8512, \
            0.8411, 0.8379, 0.8317, \
            0.8201, 0.8093, 0.8045, \
            0.9473, 0.9317, 0.9490, \
            0.9408, 0.9465, 0.9296, \
            0.9055, 0.9160, 0.9030]
allplots.append(mkbaseline) 
#all_names.append('baseline')   
all_names.append('NF+ML (M6)') 
    
AndrewNode2Vec=[0.8768, 0.8558, 0.8467, \
                0.8361, 0.5039, 0.5127, \
                0.8755, 0.6106, 0.6026, \
                0.9258, 0.9624, 0.9891, \
                0.8648, 0.5025, 0.5402, \
                0.8573, 0.6133, 0.6423]
allplots.append(AndrewNode2Vec)    
all_names.append('pure ML (M7A)') 

AndrewProNE=[0.8354, 0.8538, 0.7375, \
             0.8210, 0.7043, 0.7763, \
             0.7383, 0.7063, 0.6872, \
             0.9952, 0.9898, 0.9989, \
             0.8844, 0.9817, 0.9862, \
             0.8586, 0.8609, 0.8251]
allplots.append(AndrewProNE)    
all_names.append('pure ML (M7B)') 



Transform=[ 0.8232, 0.8253, 0.8321, \
            0.7418, 0.7659, 0.7435, \
            0.6980, 0.7023, 0.6743, \
            0.9407, 0.9373, 0.9636, \
            0.8518, 0.8804, 0.8754, \
            0.7365, 0.7977, 0.7467]
allplots.append(Transform)    
#all_names.append('Transformer')   
all_names.append('pure ML (M8)') 


fig, axs = plt.subplots(1, 2)
plt.suptitle('Prediction of new Concept Pair links', fontsize=14)

color_strs=['tab:blue','tab:orange','tab:green',
            'tab:red','tab:purple','tab:brown',
            'tab:pink','tab:cyan','tab:gray',
            'tab:olive', 'magenta']


x = np.arange(9)
for idx,pp in enumerate(allplots):
    for ii,curr_pp in enumerate(pp[0:9]):
        if ii==0:
            axs[0].hlines(curr_pp, ii, ii+1, color=color_strs[idx], label=all_names[idx])
        else:
            axs[0].hlines(curr_pp, ii, ii+1, color=color_strs[idx])
    
    for ii,curr_pp in enumerate(pp[9:]):
        if ii==0:
            axs[1].hlines(curr_pp, ii, ii+1, color=color_strs[idx], label=all_names[idx])
        else:
            axs[1].hlines(curr_pp, ii, ii+1, color=color_strs[idx])


axs[0].set_title('New single Links')
axs[1].set_title('New triple Links')

#fig.xlabel("Different Tasks")
#fig.ylabel("Area Under Curve (AUC)")
#plt.xticks(range(9))


for ii in range(2):
    axs[ii].axvline(3,color='lightgrey')
    axs[ii].axvline(6,color='lightgrey')
    axs[ii].set_ylim([0.6, 1.0])
    
    start, end = axs[ii].get_xlim()
    axs[ii].xaxis.set_ticks(np.arange(0,10, 1))
    axs[ii].set_xticklabels(['' for ii in range(10)])
    
#plt.legend(ncol=2)
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)




for ax in axs.flat:
    ax.set(xlabel='δ=1      δ=3      δ=5', ylabel='Area Under the Curve (AUC)')
    
for ax in axs.flat:
    ax.label_outer()

fig.tight_layout()
plt.savefig('highres_results_AUC.pdf', bbox_inches='tight', pad_inches=0.0, dpi=300)  # Adjust dpi value as needed

plt.show()
plt.show()