############### Effect of inter-personal relationships on polarized vs cooperative style  #####################
############### Sanjukta Krishnagopal August 2021 ###############
###################     python 3.6         ######################

import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from matplotlib.pyplot import *
random.seed(42)
import math
import numpy as np
from dbfread import DBF
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import collections
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
#import igraph as ig
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms import community as  cmty
import community
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from functools import reduce
from collections import Counter

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


######load datasets

path = 'Himalayan Database/HIMDATA/peaks.DBF'
table = DBF(path, encoding='GBK')
peaks = pd.DataFrame(iter(table))

path = 'Himalayan Database/HIMDATA/exped.csv'
expeds = pd.read_csv(path, low_memory=False)
expeds = np.array(expeds)
path = 'Himalayan Database/HIMDATA/members.csv'
members = pd.read_csv(path,sep = None, engine = 'python')


#identify all dead people
members.loc[members['DEATH,L'] == True]
peakid = members.loc[:,['PEAKID,C,4']]
members = np.array(members)
#a.columns lists all the columns


#get peak data
pk = peaks.loc[:,['PEAKID','PKNAME','HEIGHTM']]
#peaks above mean height
b = pk.loc[:,'HEIGHTM']>pk.mean()[0]
tall_pk = pk.loc[b]
b.index.values #get index values
#tall peak ID
tallpk_ID= list(tall_pk['PEAKID'])
tallpk_ht= list(tall_pk['HEIGHTM'])


#################

#compare probability of success with repeat climbing partners

names = zip(members[:,5],members[:,6], members[:,10]) #zip first, last name and age
cnt = collections.Counter(names)
ratio_success_frndz={}
ratio_fail_frndz_tired={}
ratio_fail_frndz_acc={}
ratio_fail_frndz_log={}
ratio_fail_frndz_altsick={}

#extracting number of people with more than K climbs
num_climbs = [15,20,25,30,35]
for K1 in num_climbs:
    print (K1, 'K1')
    K2 = K1+6
    multiple_names = {key : val for key, val in cnt.items() if (isinstance(val, int) and (val > K1) and (val < K2))}
    all_names = list(multiple_names.keys())
    rem_names = []
    for name in all_names:
        memberloc = np.intersect1d(np.where(members[:,5]==name[0]), np.where(members[:,6]==name[1]), np.where(members[:,10]==name[2]))
        if len(memberloc)==0:
            rem_names.append(name)
    for name in rem_names:
        all_names.remove(name)
    
    A = np.zeros((len(all_names), len(all_names) )) #holds the number of joint expeditions between person i and person j
    success = np.zeros((len(all_names), len(all_names) ))
    fail_acc = np.zeros((len(all_names), len(all_names) ))
    fail_altsick = np.zeros((len(all_names), len(all_names) ))
    fail_tired = np.zeros((len(all_names), len(all_names) ))
    fail_log = np.zeros((len(all_names), len(all_names) ))
    ind_success=[]
    ind_fail_acc=[]
    ind_fail_tired=[]
    ind_fail_log=[]
    ind_fail_altsick=[]
    
    for i in range(len(all_names)):
        print (i)
        name = all_names[i]
        memberloc = np.intersect1d(np.where(members[:,5]==name[0]), np.where(members[:,6]==name[1]), np.where(members[:,10]==name[2]))
        #get their expedition ID
        expids = members[memberloc,0].flatten()
        #calculate individuals statistics
        ci=[]
        for e in expids:
            ci.append(int(expeds[np.where(expeds[:,0] == e),30]))
        ci = np.array(ci)
        no_of_exp_i = len(ci)
        #success and fail rate of person i
        suxsi =sum( np.logical_and(ci>0, ci<4))/len(expids)
        fl_iacc = sum( ci==5)/len(expids)
        fl_ialtsick = sum( np.logical_or(ci==6, ci==8))/len(expids)
        fl_itired = sum(  ci==7)/len(expids)
        fl_ilog = sum( np.logical_or(ci==16, np.logical_and(ci>9, ci<13)))/len(expids)
        ind_success.append(suxsi)
        ind_fail_acc.append(fl_iacc)
        ind_fail_tired.append(fl_itired)
        ind_fail_log.append(fl_ilog)
        ind_fail_altsick.append(fl_ialtsick)

        #stats of expeditions with repeat partners
        for j in range(i):
            if A[j,i]==0:
                name_j = all_names[j]
                memberloc_j = np.where(members[:,5]==name_j[0]) and np.where(members[:,6]==name_j[1]) and np.where(members[:,10] == name_j[2])
                expids_j = members[memberloc_j,0]
                joint_exped = set(expids.flatten()).intersection(set(expids_j.flatten()))
                if len(joint_exped)>0:
                    A[i,j] = A[j,i]= len(joint_exped)
                else:
                    A[i,j] = A[j,i]= -1
                suxs = 0
                fl_acc = 0
                fl_tired = 0
                fl_altsick=0
                fl_log=0
                #iterate through joint expeditions of person i and j
                if len(joint_exped)>1:
                    yrs=[]
                    for e in joint_exped:
                        yrs.append(int(expeds[np.where(expeds[:,0] == e),2]))
                    minyr=np.where(yrs == min(yrs))
                    for exp in joint_exped:
                        if int(expeds[np.where(expeds[:,0] == e),2])!=minyr:
                            p = np.where(expeds[:,0] == exp)[0][-1]
                            reason = expeds[p,30]
                            if reason == 1 or reason == 2 or reason == 3:
                                suxs +=1
                            if reason == 6 or reason ==8:
                                fl_altsick +=1
                            if reason ==5:
                                fl_acc +=1
                            if reason ==7:
                                fl_tired +=1
                            if reason ==10 or reason ==11 or reason ==12 or reason ==16:
                                fl_log +=1
                    success[i,j]= success[j,i] = suxs/(len(joint_exped))
                    fail_acc[i,j]=fail_acc[j,i]=fl_acc/(len(joint_exped))
                    fail_altsick[i,j]=fail_altsick[j,i]=fl_altsick/(len(joint_exped))
                    fail_tired[i,j]=fail_tired[j,i]=fl_tired/(len(joint_exped))
                    fail_log[i,j]=fail_log[j,i]=fl_log/(len(joint_exped))
                
    #difference in success with vs without friends:
    ratio_success_frndz[K1] = []
    ratio_fail_frndz_altsick[K1] = []
    ratio_fail_frndz_log[K1] = []
    ratio_fail_frndz_tired[K1] = []
    ratio_fail_frndz_acc[K1] = []

    for i in range(len(all_names)):
        ratio_success_frndz[K1].append(np.mean(success[i,np.where(A[i]!=-1)])/ind_success[i])
        if ind_fail_altsick[i]!=0:
          ratio_fail_frndz_altsick[K1].append(np.mean(fail_altsick[i,np.where(A[i]!=-1)])/ind_fail_altsick[i])
        if ind_fail_log[i]!=0:
          ratio_fail_frndz_log[K1].append(np.mean(fail_log[i,np.where(A[i]!=-1)])/ind_fail_log[i])
        if ind_fail_tired[i]!=0:
          ratio_fail_frndz_tired[K1].append(np.mean(fail_tired[i,np.where(A[i]!=-1)])/ind_fail_tired[i])
        if ind_fail_acc[i]!=0:
          ratio_fail_frndz_acc[K1].append(np.mean(fail_acc[i,np.where(A[i]!=-1)])/ind_fail_acc[i])


climbs = ratio_success_frndz.keys()

suxs_l = [[],[]]
fail_alt_l = [[],[]]
fail_log_l = [[],[]]
fail_tired_l = [[],[]]
fail_acc_l = [[],[]]
lens_suxs = []
lens_alt = []
lens_log = []
lens_tired = []
lens_acc = []

for K in climbs:
    suxs_l[0].append(np.mean(ratio_success_frndz[K]))
    suxs_l[1].append(np.std(ratio_success_frndz[K]))
    lens_suxs.append(len(ratio_success_frndz[K]))
    fail_alt_l[0].append(np.mean(ratio_fail_frndz_altsick[K]))
    fail_alt_l[1].append(np.std(ratio_fail_frndz_altsick[K]))
    lens_alt.append(len(ratio_fail_frndz_altsick[K]))
    fail_log_l[0].append(np.mean(ratio_fail_frndz_log[K]))
    fail_log_l[1].append(np.std(ratio_fail_frndz_log[K]))
    lens_log.append(len(ratio_fail_frndz_log[K]))
    fail_tired_l[0].append(np.mean(ratio_fail_frndz_tired[K]))
    fail_tired_l[1].append(np.std(ratio_fail_frndz_tired[K]))
    lens_tired.append(len(ratio_fail_frndz_tired[K]))
    fail_acc_l[0].append(np.mean(ratio_fail_frndz_acc[K]))
    fail_acc_l[1].append(np.std(ratio_fail_frndz_acc[K]))
    lens_acc.append(len(ratio_fail_frndz_acc[K]))

xlabels = ['16-20','21-25','26-30','31-35','36-40']
figure()
plot(np.arange(len(climbs)), fail_alt_l[0], label = 'Fail_Altitude')
errorbar(np.arange(len(climbs)), fail_alt_l[0], yerr=np.divide(fail_alt_l[1], np.sqrt(lens_alt)), fmt='o',ecolor = 'red')
plot(np.arange(len(climbs)), fail_log_l[0], label = 'Fail_Logistical')
errorbar(np.arange(len(climbs)), fail_log_l[0], yerr=np.divide(fail_log_l[1], np.sqrt(lens_log)), fmt='o',ecolor = 'red')
plot(np.arange(len(climbs)), fail_tired_l[0], label = 'Fail_Fatigue')
errorbar(np.arange(len(climbs)), fail_tired_l[0], yerr=np.divide(fail_tired_l[1],np.sqrt(lens_tired)), fmt='o',ecolor = 'red')
plot(np.arange(len(climbs)), fail_acc_l[0], label = 'Fail_Accident')
errorbar(np.arange(len(climbs)), fail_acc_l[0], yerr=np.divide(fail_acc_l[1],np.sqrt(lens_acc)), fmt='o',ecolor = 'red')
legend(loc = 2, framealpha=0.5)
ylim(0,1.1)
xticks(np.arange(len(climbs)), xlabels, rotation=0, fontsize = 12)
xlabel('Total climbs', fontsize = 14)
ylabel(r'$\frac{avg \; failure \; w \; repeat \; partners}{overall \; avg \; failure}$', fontsize = 14)
savefig('Effect_of_company.pdf',bbox_inches='tight')

                

#extract expeds and its people from everest
ev_pk_no = np.where(expeds[:,1]=='EVER')[0]

#remove incomplete expeditions, and ones with less than 12 people
success = np.where(np.logical_and(list(expeds[:,30]<4), list(expeds[:,30]>0)))[0]
no_mem_few= np.where(list(expeds[:,38]>11))[0]
d = [success,no_mem_few,ev_pk_no]
exp_nums_ev = list(set.intersection(*[set(x) for x in d])) #indices of the selected expeditions
ev_exp_no = expeds[exp_nums_ev,0]

#contains all data of the members in each everest expedition listed in ev_exp_no
mem_ever={}
for e in ev_exp_no:
    mem_exp = np.where(members[:,0]==e)[0]
    mem_ever[e] = members[mem_exp]


#find simplicial degree of each person

success = []
prev_exper = []
suc_frac_persist = []
suc_frac_persist_exper=[]
exp_clique_style = []
figure()
mult_th_vals = [1,2,3,4]
pearson = []
max_sim_dim_all = []
nonz_exp=0
for mult_th in mult_th_vals: #weight threshold
    exp_success = []
    simp_deg = []
    suc_count_th=[]
    fail_count_th=[]
    ded_count_th=[]
    exp_mean_simpl_deg=[]
    exp_var_simpl_deg=[]
    
    for e in ev_exp_no:
        prev_exped = {}
        success_member = np.zeros(len(mem_ever[e]))
        all_prev_exped=[]
        exper_member = np.zeros(len(mem_ever[e]))
        #get list of all previous exp ids of all people in this expedition
        for i in range(len(mem_ever[e])):
            per = mem_ever[e][i]
            prev_e = reduce(np.intersect1d,(np.where(members[:,5]==per[5])[0], np.where(members[:,6]==per[6])[0], np.where(members[:,10]==per[10])[0],np.where(members[:,3]<per[3])[0]))
            prev_exped[i] = list(members[prev_e,0]) #exp_ids_of previous expeditions of this person
            exper_member[i] = len(prev_e)
            if per[25] == True:
                success_member[i]=1
            if per[54] == True:
                success_member[i]=-1 #death        
            all_prev_exped.extend(prev_exped[i]) 
        all_prev_exped = list(set(all_prev_exped))       
        #identify the max simplicial degree of all people in the expedition
        max_deg = {}
        all_simplexes = collections.defaultdict(list)
        for i in range(len(mem_ever[e])):
            max_deg[i] = 0
        #initialize    
        exp_content = {}
        for exp in all_prev_exped:
            i = np.where(expeds[:,0]==exp)[0]
        #loop
        for i in range(len(mem_ever[e])):
            exp_name = []
            for ee in prev_exped[i]:
                count_ = np.zeros(len(mem_ever[e])) 
                simplex = []
                simplex.append(i)
                for j in range(i):
                    if ee in prev_exped[j]:
                        count_[i]+=1 #degree of simplex 
                        simplex.append(j)                   
                for k in simplex:
                    count_[k]=count_[i]
                if count_[k]>max_deg[k]:
                    max_deg[k] = count_[k]
                if count_[i]>max_deg[i]:
                    max_deg[i] = count_[i]
                all_simplexes[tuple(np.sort(simplex))].append(ee)
                exp_name.append(ee)
        c= max([len(i) for i in prev_exped.values()])#max possible multiplity of a simplex
        
        ##### store ########
        count_thresh = np.zeros((max(c,mult_th),len(mem_ever[e]),int(max(list(max_deg.values())))+1)) #the first index tells you what the threhsold for simplex occurence is, second is member, third is their max_degree with that threshold
        for sim in all_simplexes.keys():
            if len(sim)>1:
                occ = len(set(all_simplexes[sim]))
                for j in sim:
                    count_thresh[occ-1,j,len(sim)-1]+=1       
        #find max degree for different thresholds
        success.extend(success_member)
        prev_exper.extend(exper_member)
        
        #calculate matrix
        f = count_thresh[mult_th-1:] 
        f = np.swapaxes(f,0,1)         
        max_degree = []
        for row in f: #iterate over individuals
            max_degree_th = []
            for th in row: #iterate over simplicial weights above threshold 
                if max(th)==0:
                    max_degree_th.append(0)
                else:
                    max_degree_th.append(th.nonzero()[0].max())
            max_degree.append(max(max_degree_th))
        simp_deg.extend(max_degree)
        s = np.where(expeds[:,0]== e)[0][0]
        exp_success.append(expeds[s,39]/expeds[s,38])
        exp_mean_simpl_deg.append(np.mean(max_degree))
        exp_var_simpl_deg.append(np.var(max_degree))
        
        #calculate success fraction amongst people not part of the clique   
        if max(max_degree)>0 and mult_th == 1:
            nonz_exp+=1     
            max_clique = np.where(max_degree==max(max_degree))[0][0] 
            wt = np.argmax(count_thresh[:, max_clique, max(max_degree)])+1
            other = np.where(max_degree!=max(max_degree))[0]           
            success_other = success_member[other]
            success_other[success_other==-1.0] = 0
            exp_clique_style.append([max(max_degree),wt, np.mean(success_other)])
            
    ## successful individuals vs their max simpl degree across various thresholds
    clmbr_nonzero = np.where(np.array(simp_deg)!=0)[0]
    print ('number of climbers', len(clmbr_nonzero), mult_th)
    success_ = np.array(success)[clmbr_nonzero]
    prev_exper_=np.array( prev_exper)[clmbr_nonzero]
    simp_deg_ = np.array(simp_deg)[clmbr_nonzero]

    #log
    clmbr_s = np.where(np.array(success_).astype(int)==1)[0] 
    suc_frac_persist.append(len(clmbr_s)/len(success_)) 
    clmbr_f = np.where(np.array(success_).astype(int)==0)[0]
    suc_count = list(Counter(simp_deg_[clmbr_s]).values())[1:]
    fail_count =  list(Counter(simp_deg_[clmbr_f]).values())[1:]
    clmbr_d = np.where(np.array(success_).astype(int)==-1)[0] 
    ded_count =  list(Counter(simp_deg_[clmbr_f]).values())[1:]
    ded_count_th.append(ded_count)
    suc_count_th.append(suc_count)
    fail_count_th.append(fail_count)    

    figure()
    clmbr_s_nonzero = simp_deg_[clmbr_s]
    w_s = 1/len(clmbr_s_nonzero)
    clmbr_f_nonzero = simp_deg_[clmbr_f]
    w_f = 1/len(clmbr_f_nonzero)
    clmbr_d_nonzero = simp_deg_[clmbr_d]
    w_d = 1/len(clmbr_d_nonzero)
    max_sim_dim = max(max(clmbr_s_nonzero),max(clmbr_f_nonzero),max(clmbr_d_nonzero) )
    max_sim_dim_all.append(max_sim_dim)
    bins = np.arange(1,max_sim_dim+2)-0.5
    hist([clmbr_s_nonzero,clmbr_f_nonzero, clmbr_d_nonzero], weights = [[w_s]*len(clmbr_s_nonzero),[w_f]*len(clmbr_f_nonzero),[w_d]*len(clmbr_d_nonzero) ], bins = bins, rwidth = 0.8, label = ['Summit success','No summit', 'Death'])
    legend(loc=1,framealpha=0.7)
    xlabel('Influence (highest simplicial dimension)', fontsize = 14)
    ylabel('Fraction of climbers', fontsize = 14)
    xticks(range(1, max_sim_dim+1))
    savefig('hist_th_'+str(mult_th)+'.pdf')

    ####plotting###
    exp_nonzero = np.where(np.array(exp_mean_simpl_deg)!=0.0)[0] 
    print ('number of nonzero expeds', len(exp_nonzero), mult_th)
    exp_success = np.array(exp_success)[exp_nonzero]
    figure()
    scatter(np.array(exp_mean_simpl_deg)[exp_nonzero], exp_success)
    ylabel('Expedition Success', fontsize =14)
    xlabel('Average simplicial dimension', fontsize =14)
    savefig('exp_success_vs_exp_dim'+str(mult_th)+'.pdf', bbox_inches = 'tight')
    pear = pearsonr(np.array(exp_var_simpl_deg)[exp_nonzero], exp_success)
    pearson.append(pear)
    


#calculate avg success rate of others vs max simplicial degree:
exp_clique_style = np.array(exp_clique_style) # [columns are: largest dim of clique, weight, success of people not part of clique]
ind_deg=np.unique(exp_clique_style[:,0])
suc_deg={}
for i in ind_deg:
    suc_deg[i] = [np.mean(exp_clique_style[np.where(exp_clique_style[:,0]==i)[0],2]),np.var(exp_clique_style[np.where(exp_clique_style[:,0]==i)[0],2])]
    
figure()
scatter(suc_deg.keys(), np.array(list(suc_deg.values()))[:,0])
text(0.1,0.35, 'polarized', size = 14)
text(12,0.35, 'cooperative', size=14)
xlim(0,15.5)
xlabel('Largest simplicial dimension', fontsize = 14)
ylabel('Success rate of non-simplex members', fontsize = 14)
savefig('avg_success_rate_of_outsiders_of_clique.pdf',bbox_inches='tight')    



###############importance of personal features ###############

#create a network for each expedition:
exp_adj = {}
#nodes: calculated age, sex, sherpa, oxygen ascending/ oxygen descending, previous experience in climbing peaks more than 8000m
num_var = 6

#centrality for succesful climbers, fail (unsuccesful) climbers, and all
centrality_S= []
centrality_F= []
centrality_A= []
nofail_count=0
count=0
for exp in ev_exp_no:
    print (count)
    S = np.zeros((num_var, num_var)) #successful folks
    U = np.zeros((num_var, num_var)) #unsuccessful folks
    #member list
    climbr = mem_ever[exp]
    succ = np.where(np.array(climbr)[:,25]==True)[0]
    fail = np.where(np.array(climbr)[:,25]==False)[0]
    for s in range(len(climbr)):
        climbr_data = np.zeros(num_var)
        if climbr[s][7] =='M': #men
            climbr_data[0] = 1
        if climbr[s][11] >=40:#age
            climbr_data[1] = 1
        if climbr[s][23]==True: #sherpa
            climbr_data[2] = 1
        if climbr[s][49]==True: #oxygen ascending
            climbr_data[3] = 1
        if climbr[s][50]==True: #oxygen descending
            climbr_data[4] = 1
        #now to check if they have any previous experience above 8000m
        other = np.where(np.logical_and.reduce((list(members[:,5]==climbr[s][5]), list(members[:,6]==climbr[s][6]), list(members[:,3]<climbr[s][3]))))[0]
        if (members[other,34]).any()>=8000:
            climbr_data[5] = 1 #has previous experience above 8000m
        if s in succ:
            S+=np.outer(climbr_data,climbr_data) #avg adjacency matrix of succesful climbers in the expedition
        if s in fail:
            U+=np.outer(climbr_data,climbr_data)
    All = (S+U)/len(climbr)
    if len(climbr)==0:
        print (exp, 'wtf')
    S/=len(succ)
    if len(fail)==0:
        nofail_count+=1
    else:
        U/=len(fail)
    exp_adj[exp]=[S,U,All, len(succ)/len(climbr)] #last one is fraction of team that is succesful
    if len(S)!=0:
        central = nx.eigenvector_centrality(nx.from_numpy_matrix(S)) #degree or eigenvector centrality
        centrality_S.append(list(central.values()))
    if len(U)!=0:
        central = nx.eigenvector_centrality(nx.from_numpy_matrix(U)) #degree or eigenvector centrality
        centrality_F.append(list(central.values()))
    if len(All)!=0:
        central = nx.eigenvector_centrality(nx.from_numpy_matrix(All)) #degree or eigenvector centrality
        centrality_A.append(list(central.values()))
    count+=1


#node centrality in succesful people
centrality_S=np.array(centrality_S)
centrality_F=np.array(centrality_F)
centrality_A=np.array(centrality_A)

centrality = [x/np.sum(x) for x in centrality_S]
centrals_S=np.mean(centrality, axis=0)
centrals_std_S = np.var(centrality, axis=0) #/np.sqrt(len(centrality))

centrality = [x/np.sum(x) for x in centrality_A]
centrals_A=np.mean(centrality, axis=0)
centrals_std_A = np.var(centrality, axis=0) #/np.sqrt(len(centrality))

centrality = [x/np.sum(x) for x in centrality_F]
centrals_F=np.mean(centrality, axis=0)
centrals_std_F = np.var(centrality, axis=0) #/np.sqrt(len(centrality))

po=np.argsort(np.abs(centrals_S-centrals_F))
xlabels = ['sex(M)', 'age(>40)', 'sherpa', 'O2asc', 'O2desc', 'exp>8000m']

###plot ###
#plot centrality for succesful and unsuccesful
figure()
plot(np.arange(len(centrals_S)), centrals_S[pos], label = 'Summit Success', color = 'blue')
errorbar(np.arange(len(centrals_S)), centrals_S[pos], yerr=centrals_std_S[pos], fmt='o',ecolor = 'red')
plot(np.arange(len(centrals_F)), centrals_F[pos], label = 'No Summit', color = 'brown')
errorbar(np.arange(len(centrals_F)), centrals_F[pos], yerr=centrals_std_F[pos], fmt='o',ecolor = 'red')
legend(loc = 2, fontsize = 14)
xticks(np.arange(len(centrals_S)), np.array(xlabels)[pos], rotation=0, fontsize = 12)
yticks(fontsize=13)
xlabel('Personal factors', fontsize = 14)
ylabel('Centrality', fontsize = 14)
savefig('Centrality_plots_.pdf',bbox_inches='tight')




#######Identify correlaion of expedition-wide factors  with expedition success rate

c1 = [exp_adj[expid_i][2] for expid_i in ev_exp_no]
c2 = expeds[exp_nums_ev,27] 
c3 = expeds[exp_nums_ev,36]
c4 = expeds[exp_nums_ev,38]
c5 = []
for i in range(len(ev_exp_no)):
    crowdi = np.where( np.logical_and(expeds[:,25] == expeds[exp_nums_ev[i],25],expeds[:,1] == expeds[exp_nums_ev[i],1]))[0]
    c5.append(np.sum(expeds[crowdi,39]) + np.sum(expeds[crowdi,42]))

suc_rate = (expeds[exp_nums_ev,39]+ expeds[exp_nums_ev,42])/(expeds[exp_nums_ev,38] + expeds[exp_nums_ev,41])

#extract only unique entries (upper triangular entries of l1)
utri = np.triu_indices(np.shape(c1[0])[0],1)
c1 = [s[utri] for s in c1]

reg_ = LinearRegression(fit_intercept = True).fit(c1, suc_rate)
print ('layer coefficient for success', reg_.coef_, reg_.intercept_)

c1_out = np.dot(c1,reg_.coef_)

#finding correlation coefficient
pc1 = pearsonr(c1_out, suc_rate)
pc2 = pearsonr(c2, suc_rate)
pc3 = pearsonr(c3, suc_rate)
pc4 = pearsonr(c4, suc_rate)
pc5 = pearsonr(c5, suc_rate)

#plot
figure()
plot(np.arange(4), [pc2[0],pc3[0],pc4[0], pc5[0]], marker ='o')
xticks(np.arange(4),[])
xlim(-0.5,3.5)
annotate('Days to high point', (0-0.5,pc2[0]), fontsize = 13)
annotate('#Camps > basecamp', (1-0.5,pc3[0]), fontsize = 13)
annotate('#Members/#hired personnel', (2-0.5,pc4[0]), fontsize = 13)
annotate('Exp size', (3-0.4,pc5[0]), fontsize = 13)
#annotate('Intra-exp climber features', (3-1.5,pc1[0]), fontsize = 13)
ylabel('PCC with exp success rate', fontsize = 14)
xlabel('Expedition-wide factors', fontsize = 14)
yticks(fontsize=12)
savefig('pcc_layer_summit.pdf',bbox_inches='tight')


