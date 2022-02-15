import torch
from torch import nn
import random, time
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy import sparse
import pickle

from utils import create_training_data, create_training_data_biased, calculate_ROC, NUM_OF_VERTICES

class ff_network(nn.Module):

    def __init__(self):
        """
        Fully Connected layers
        """
        super(ff_network, self).__init__()

        self.semnet = nn.Sequential( # very small network for tests
            nn.Linear(15, 100), # 15 properties
            nn.ReLU(),
            nn.Linear(100, 100), 
            nn.ReLU(),       
            nn.Linear(100, 10),
            nn.ReLU(),             
            nn.Linear(10, 1)
        )


    def forward(self, x):
        """
        Pass throught network
        """
        res = self.semnet(x)

        return res


def train_model(model_semnet, data_train0, data_train1, data_test0, data_test1, lr_enc, batch_size, data_source):
    """
    Training the neural network
    """            
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size_of_loss_check=2000
    
    optimizer_predictor = torch.optim.Adam(model_semnet.parameters(), lr=lr_enc)
    
    data_train0=torch.tensor(data_train0, dtype=torch.float).to(device)
    data_test0=torch.tensor(data_test0, dtype=torch.float).to(device)
    
    data_train1=torch.tensor(data_train1, dtype=torch.float).to(device)
    data_test1=torch.tensor(data_test1, dtype=torch.float).to(device)

    test_loss_total=[]
    moving_avg=[]
    criterion = torch.nn.MSELoss()
    
    # There are much more vertex pairs that wont be connected (0) rather than ones
    # that will be connected (1). However, we observed that training with an equally weighted
    # training set (same number of examples for (0) and (1)) results in more stable training.
    # (Imaging we have 1.000.000 nonconnected and 10.000 connected)
    #
    # For that reason, we dont have true 'episodes' (where each example from the training set
    # has been used in the training). Rather, in each of our iteration, we sample batch_size
    # random training examples from data_train0 and from data_train1.
    
    for iteration in range(50000): # should be much larger, with good early stopping criteria
        model_semnet.train()
        data_sets=[data_train0,data_train1]
        total_loss=0
        for idx_dataset in range(len(data_sets)):
            idx = torch.randint(0, len(data_sets[idx_dataset]), (batch_size,))
            data_train_samples = data_sets[idx_dataset][idx]
            calc_properties = model_semnet(data_train_samples)
            curr_pred=torch.tensor([idx_dataset] * batch_size, dtype=torch.float).to(device)
            real_loss = criterion(calc_properties, curr_pred)
            total_loss += torch.clamp(real_loss, min = 0., max = 50000.).double()

        optimizer_predictor.zero_grad()
        total_loss.backward()
        optimizer_predictor.step()

        # Evaluating the current quality.
        with torch.no_grad():
            model_semnet.eval()
            # calculate train set
            eval_datasets=[data_train0,data_train1,data_test0,data_test1]
            all_real_loss=[]
            for idx_dataset in range(len(eval_datasets)):
                eval_datasets[idx_dataset]
                calc_properties = model_semnet(eval_datasets[idx_dataset][0:size_of_loss_check])        
                curr_pred=torch.tensor([idx_dataset%2] * len(eval_datasets[idx_dataset][0:size_of_loss_check]), dtype=torch.float).to(device)
                real_loss = criterion(calc_properties, curr_pred)
                all_real_loss.append(real_loss.detach().cpu().numpy())
             
            test_loss_total.append(np.mean(all_real_loss[2])+np.mean(all_real_loss[3]))

            if iteration%50==0:
                info_str='iteration: '+str(iteration)+' - train loss: '+str(np.mean(all_real_loss[0])+np.mean(all_real_loss[1]))+'; test loss: '+str(np.mean(all_real_loss[2])+np.mean(all_real_loss[3]))
                print('    train_model: ',info_str)
                with open("logs_"+data_source+".txt", "a") as myfile:
                    myfile.write('\n    train_model: '+info_str)

            if len(test_loss_total)>200: # early stopping
                test_loss_moving_avg=sum(test_loss_total[-100:])
                moving_avg.append(test_loss_moving_avg)
                if len(moving_avg)>10:
                    if moving_avg[-1]>moving_avg[-5] and moving_avg[-1]>moving_avg[-25]:
                        print('    Early stopping kicked in')
                        break

    plt.plot(test_loss_total)
    plt.show()
    
    plt.plot(test_loss_total[500:])
    plt.show()    
    
    plt.plot(moving_avg)
    plt.show()

    return True
    

def compute_all_properties(all_sparse,AA02,AA12,AA22,all_degs0,all_degs1,all_degs2,all_degs02,all_degs12,all_degs22,v1,v2):
    """
    Computes hand-crafted properties for one vertex in vlist
    """
    all_properties=[]

    all_properties.append(all_degs0[v1]) # 0
    all_properties.append(all_degs0[v2]) # 1
    all_properties.append(all_degs1[v1]) # 2
    all_properties.append(all_degs1[v2]) # 3
    all_properties.append(all_degs2[v1]) # 4
    all_properties.append(all_degs2[v2]) # 5
    all_properties.append(all_degs02[v1]) # 6
    all_properties.append(all_degs02[v2]) # 7
    all_properties.append(all_degs12[v1]) # 8
    all_properties.append(all_degs12[v2]) # 9
    all_properties.append(all_degs22[v1]) # 10
    all_properties.append(all_degs22[v2]) # 11

    all_properties.append(AA02[v1,v2]) # 12
    all_properties.append(AA12[v1,v2]) # 13
    all_properties.append(AA22[v1,v2]) # 14    

    return all_properties



def compute_all_properties_of_list(all_sparse,vlist,data_source):
    """
    Computes hand-crafted properties for all vertices in vlist
    """
    time_start=time.time()
    AA02=all_sparse[0]**2
    AA02=AA02/AA02.max()
    AA12=all_sparse[1]**2
    AA12=AA12/AA12.max()
    AA22=all_sparse[2]**2
    AA22=AA22/AA22.max()
    
    all_degs0=np.array(all_sparse[0].sum(0))[0]
    if np.max(all_degs0)>0:
        all_degs0=all_degs0/np.max(all_degs0)
        
    all_degs1=np.array(all_sparse[1].sum(0))[0]
    if np.max(all_degs1)>0:
        all_degs1=all_degs1/np.max(all_degs1)
    
    all_degs2=np.array(all_sparse[2].sum(0))[0]
    if np.max(all_degs2)>0:
        all_degs2=all_degs2/np.max(all_degs2)

    all_degs02=np.array(AA02[0].sum(0))[0]
    if np.max(all_degs02)>0:
        all_degs02=all_degs02/np.max(all_degs02)
        
    all_degs12=np.array(AA12[1].sum(0))[0]
    if np.max(all_degs12)>0:
        all_degs12=all_degs12/np.max(all_degs12)
        
    all_degs22=np.array(AA22[2].sum(0))[0]
    if np.max(all_degs22)>0:
        all_degs22=all_degs22/np.max(all_degs22)
    
    all_properties=[]
    print('    Computed all matrix squares, ready to ruuuumbleeee...')
    for ii in range(len(vlist)):
        vals=compute_all_properties(all_sparse,
                                    AA02,
                                    AA12,
                                    AA22,
                                    all_degs0,
                                    all_degs1,
                                    all_degs2,
                                    all_degs02,
                                    all_degs12,
                                    all_degs22,
                                    vlist[ii][0],
                                    vlist[ii][1])

        all_properties.append(vals)
        if ii%10**4==0:
            print('    compute_all_properties_of_list progress: (',time.time()-time_start,'sec) ',ii/10**6,'M/',len(vlist)/10**6,'M')

            with open("logs_"+data_source+".txt", "a") as myfile:
                myfile.write('\n    compute_all_properties_of_list progress: ('+str(time.time()-time_start)+'sec) '+str(ii/10**6)+'M/'+str(len(vlist)/10**6)+'M')

            time_start=time.time()

    return all_properties


def flatten(t):
    return [item for sublist in t for item in sublist]


def link_prediction_semnet(full_dynamic_graph_sparse,unconnected_vertex_pairs,year_start,years_delta,vertex_degree_cutoff,min_edges,hyper_parameters,data_source):
    """
    Gets an evolving semantic network and a list of unconnected vertices,
    and returns an index list of which vertex pairs are most likely to be
    connected at a later time t2 (from likely to unlikely)
    
    :param full_dynamic_graph_sparse: Full graph, numpy array dim(n,3)
            [vertex v1, vertex v2, time stamp t1] representing edges, up
            to t<=t1. The edge is formed between vertex v1 and v2 at time t
            (measured in days after 1.1.1990)
    :param unconnected_vertex_pairs, numpy array of vertex
            pairs [v1,v2] with no edge at t1 and deg(v1/2)>10. Question is
            whether these vertex pairs will have an edge at t2.
    :param year_start - Integer, year for t1. Edges with
            t1=(date(year_start,12,31)-date(1990,1,1)).days are included
            in full_dynamic_graph_sparse
    :param years_delta, Integer, number if years to predict,
            t2=(date(year_start+years_delta,12,31)-date(1990,1,1)).days
    :param vertex_degree_cutoff, Integer, number of minimal vertex degree for
            prediction 
    :param min_edges, Integer, Predict edges which grew from zero to min_edges 
        
    Output - sorted_predictions_eval, numpy array with
            len(..)=len(unconnected_vertex_pairs).
            Sorted of which pairs in unconnected_vertex_pairs
            are most likely to be connected at t2. Used for computing the
            AUC metric.

    
    This is a simple baseline model, with the following workflow:
    1) Learns to predict using training data from 2011 -> 2014.
          1.1) For that, it uses 
                     train_dynamic_graph_sparse,
                     train_edges_for_checking,
                     train_edges_solution=
                     create_training_data(
                                          full_graph,
                                          year_start=2014,
                                          years_delta=3,
                                          edges_used,
                                          vertex_degree_cutoff
                                         )
               train_dynamic_graph_sparse - Semantic network until 2014
               (numpy array with triples for each edge [v1,v2,t])
               train_edges_for_checking - list of unconnected vertices in
               2011 and computes whether they are connected by 2014.
                   
               edges_used, unconnected edges that are used in training.
               
               train_edges_solution is a numpy array stating whether an
               element in edges_used has been connected in t2
               
          1.2) It computes a list of 15 properties of each edge in the
               train_edges_for_checking. The properties contain the
               local degrees of the vertices, numbers of shared neighbors
               and paths of length 3, in the year 2011, 2010 and 2009.
          1.3) Those 15 properties per vertex pair are input into a neural
               network, which predicts whether the vertex pairs will be
               connected or not (using train_edges_solution)
          1.4) Computes the AUC for training and test data using
               calculate_ROC.
    
    2) Makes predictions for 2014 -> 2017 data.
          2.1) Computes the 15 properties for the 2014 data.
          2.2) Uses the trained network to predict whether edges are
               created by 2017.
    
    3) Creates a sorted index list, from highest predicted vertex pair to
        least predicted one (sorted_predictions)
    
    4) Returns sorted_predictions
    """

    edges_used, percent_positive_examples, batch_size, lr_enc, rnd_seed=hyper_parameters

    random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)
    np.random.seed(rnd_seed)
    
    print('1) Learns to predict using training data from '+str(year_start-years_delta)+' -> '+str(year_start))
    
    print('1.1) Create training data for '+str(year_start-years_delta))

    with open("logs_"+data_source+".txt", "a") as myfile:
        myfile.write('1.1) Create training data for '+str(year_start-years_delta)+'\n') 
    
    
    train_dynamic_graph_sparse,train_edges_for_checking,train_edges_solution = create_training_data_biased(full_dynamic_graph_sparse, year_start-years_delta, years_delta, min_edges=min_edges, edges_used=edges_used, vertex_degree_cutoff=vertex_degree_cutoff, data_source=data_source)

    day_origin = date(1990,1,1)
    years=[year_start-years_delta,year_start-years_delta-1,year_start-years_delta-2]

    train_sparse=[]
    for yy in years:
        print('    Create Graph for ', yy)
        day_curr=date(yy,12,31)
        train_edges_curr=train_dynamic_graph_sparse[train_dynamic_graph_sparse[:,2]<(day_curr-day_origin).days]
        adj_mat_sparse_curr = sparse.csr_matrix((np.ones(len(train_edges_curr)), (train_edges_curr[:,0], train_edges_curr[:,1])), shape=(NUM_OF_VERTICES,NUM_OF_VERTICES))
    
        train_sparse.append(adj_mat_sparse_curr)
    
    
    print('    Shuffle training data...')
    with open("logs_"+data_source+".txt", "a") as myfile:
        myfile.write('\n    Shuffle training data...\n')     
    train_valid_test_size=[0.9, 0.1, 0.0]
    x = [i for i in range(len(train_edges_for_checking))]  # random shuffle input

    random.shuffle(x)
    train_edges_for_checking = train_edges_for_checking[x]
    train_edges_solution = train_edges_solution[x]

    print('    Split dataset...')
    idx_traintest=int(len(train_edges_for_checking)*train_valid_test_size[0])

    data_edges_train=train_edges_for_checking[0:idx_traintest]
    solution_train=train_edges_solution[0:idx_traintest]  

    data_edges_test=train_edges_for_checking[idx_traintest:]
    solution_test=train_edges_solution[idx_traintest:]


    print('1.2) Compute 15 network properties for training data y=',year_start-3)

    print('    Prepare data for equally distributed training...')
    print('    This is an important design choice for training the NN.')
    print('    Note that the evaluation set (also for the competition) is NOT equally distributed!')
    with open("logs_"+data_source+".txt", "a") as myfile:
        myfile.write('\n1.2) Compute 15 network properties for training data y='+str(year_start-3))   

    # Rather than using all connected and unconnected vertex pairs for training
    # (i.e. needing to compute their properties), we reject about 99% of all unconnected
    # examples, to have more examples of connected cases in the training. This significantly
    # speeds up the computation, at the price of precision.
    data_edges_train_smaller=[]
    solution_train_smaller=[]
    for ii in range(len(data_edges_train)):
        if (solution_train[ii]==0 and random.random()<percent_positive_examples) or solution_train[ii]==1:
            data_edges_train_smaller.append(data_edges_train[ii])
            solution_train_smaller.append(solution_train[ii])
    
    with open("logs_"+data_source+".txt", "a") as myfile:
        myfile.write('\nComputing properties for Training data')   
    print('Computing properties for Training data')
    data_train=compute_all_properties_of_list(train_sparse,data_edges_train_smaller,data_source)


    data_train0=[]
    data_train1=[]
    for ii in range(len(data_edges_train_smaller)):
        if solution_train_smaller[ii]==1:
            data_train1.append(data_train[ii])
        else:
            data_train0.append(data_train[ii])


    with open("logs_"+data_source+".txt", "a") as myfile:
        myfile.write('\nComputing properties for Test data')   
    print('Computing properties for Test data')
    data_test=compute_all_properties_of_list(train_sparse,data_edges_test,data_source)
    data_test0=[]
    data_test1=[]
    for ii in range(len(data_edges_test)):
        if solution_test[ii]==1:
            data_test1.append(data_test[ii])
        else:
            data_test0.append(data_test[ii])

    with open("logs_"+data_source+".txt", "a") as myfile:
        myfile.write('\n1.3) Train Neural Network')   
    print('1.3) Train Neural Network')      
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_semnet = ff_network().to(device)

    model_semnet.train()
    train_model(model_semnet, data_train0, data_train1, data_test0, data_test1, lr_enc, batch_size, data_source)    
    
    with open("logs_"+data_source+".txt", "a") as myfile:
        myfile.write('\n1.4) Computes the AUC for training and test data using calculate_ROC.')       
    print('1.4) Computes the AUC for training and test data using calculate_ROC.')
    model_semnet.eval()
    
    data_train=torch.tensor(data_train, dtype=torch.float).to(device)
    all_predictions_train=flatten(model_semnet(data_train).detach().cpu().numpy())
    sorted_predictions_train=np.flip(np.argsort(all_predictions_train,axis=0))    
    AUC_train=calculate_ROC(sorted_predictions_train, solution_train_smaller)
    print('    AUC_train: ', AUC_train)
    
    data_test=torch.tensor(data_test, dtype=torch.float).to(device)
    all_predictions_test=flatten(model_semnet(data_test).detach().cpu().numpy())
    sorted_predictions_test=np.flip(np.argsort(all_predictions_test,axis=0))    
    AUC_test=calculate_ROC(sorted_predictions_test, solution_test)
    print('    AUC_test: ', AUC_test)

    # Create properties for evaluation
    print('2) Makes predictions for '+str(year_start)+' -> '+str(year_start+years_delta)+' data.')
    years=[year_start,year_start-1,year_start-2]

    with open("logs_"+data_source+".txt", "a") as myfile:
        myfile.write('\n2.1) Computes the 15 properties for the '+str(year_start)+' data.')    
    print('2.1) Computes the 15 properties for the '+str(year_start)+' data.')
    eval_sparse=[]
    for yy in years:
        print('    Create Graph for ', yy)
        day_curr=date(yy,12,31)
        eval_edges_curr=full_dynamic_graph_sparse[full_dynamic_graph_sparse[:,2]<(day_curr-day_origin).days]
        adj_mat_sparse_curr = sparse.csr_matrix(
                                                (np.ones(len(eval_edges_curr)), (eval_edges_curr[:,0], eval_edges_curr[:,1])),
                                                shape=(NUM_OF_VERTICES,NUM_OF_VERTICES)
                                               )

        eval_sparse.append(adj_mat_sparse_curr)

    with open("logs_"+data_source+".txt", "a") as myfile:
        myfile.write('\n    compute all properties for evaluation')    
    print('    compute all properties for evaluation')
    eval_examples=compute_all_properties_of_list(eval_sparse,unconnected_vertex_pairs,data_source)
    eval_examples=np.array(eval_examples)
    
    with open("logs_"+data_source+".txt", "a") as myfile:
        myfile.write('\n2.2) Uses the trained network to predict whether edges are created by '+str(year_start+3)+'.')        
    print('2.2) Uses the trained network to predict whether edges are created by '+str(year_start+3)+'.')
    eval_examples=torch.tensor(eval_examples, dtype=torch.float).to(device)
    all_predictions_eval=flatten(model_semnet(eval_examples).detach().cpu().numpy())

    with open("logs_"+data_source+".txt", "a") as myfile:
        myfile.write('\n3) Creates a sorted index list, from highest predicted vertex pair to least predicted one (sorted_predictions)')    
    print('3) Creates a sorted index list, from highest predicted vertex pair to least predicted one (sorted_predictions)')
    sorted_predictions_eval=np.flip(np.argsort(all_predictions_eval,axis=0))   
       
    print('4) Returns sorted_predictions')
    return sorted_predictions_eval
