import numpy as np
import pickle


def get_embed_dict(embed_file, embed_dict_file, input_dim):
    embed_dict = {}
    with open(embed_file, 'r') as f:
        for line in f:
            entity_embed = line.rstrip('\n').split(' ')
            if len(entity_embed)-1!=int(input_dim):
                continue
            embed_dict[entity_embed[0]] = np.array(entity_embed[1:], dtype=float)

    print("Saving embedding dict to " + embed_dict_file + ".pkl...")
    pickle.dump(embed_dict,open(embed_dict_file + '.pkl', "wb"))


def get_field_file(train_path, field_file):
    new_data = []
    field_data = {}
    with open(train_path,'r') as f:
        for line in f:
            tmp = line.split(' ')
            if tmp[0] not in field_data.keys():
                field_data[tmp[0]] = 1
            if tmp[1] not in field_data.keys():
                field_data[tmp[1]] = 1
            new_data.append('s_'+tmp[0]+" d_"+tmp[1]+" "+tmp[2])
            new_data.append('s_'+tmp[1]+" d_"+tmp[0]+" "+tmp[2])

    with open(field_file,'w') as f:
        for key,value in field_data.items():
            f.write(str(key)+" s\n")

