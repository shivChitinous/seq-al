# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def global_dp(seq_1,seq_2,S,g,N=10,max_i=None,disp_scoring=False,disp_tracer=False,
              disp_alignments=False,high_low=False):
    
    #set max iterations
    if max_i is None: max_i = N**2
    
    #initialize the matrix
    m = len(seq_1); n = len(seq_2)
    M = np.zeros([m+1,n+1])
    M[0,:] = g*np.arange(0,n+1)
    M[:,0] = g*np.arange(0,m+1)

    #fill up    
    tracer = np.zeros([np.shape(M)[0],np.shape(M)[1],3])
    for i in range(1,m+1):
        for j in range(1,n+1):
            arr = np.array([M[i-1,j-1]+(S[seq_1[i-1]][seq_2[j-1]]),(M[i-1,j]+g),(M[i,j-1]+g)])
            M[i,j] = np.max(arr)
            idx = np.where(arr==np.max(arr))
            tracer[i,j,idx] = 1
    
    #traceback
    alignment = traceback_iterator(tracer,seq_1,seq_2,N=N,max_i=max_i,high_low=high_low)
    
    if disp_alignments is True:
        print("Max. score = "+str(M[-1,-1]))
        for i,e in enumerate(alignment):
            print(str(i+1)+".","\n",e[0],"\n",e[1],"\n")
    
    if disp_scoring is True:
        plt.figure(figsize = (5,5))
        sns.heatmap(M,linecolor='white',linewidth=1,cmap="viridis",square=True)
    
    if disp_tracer is True:
        fig, ax = plt.subplots(1,3,figsize = (12,4))
        for i,p in enumerate([r"$\nwarrow$",r"$\uparrow$",r"$\leftarrow$"]):
            sns.heatmap(tracer[:,:,i],linecolor='white',linewidth=1,
                        cmap="coolwarm",vmin=-0.5,vmax=0.6,square=True,cbar=False,ax=ax[i]);
            ax[i].set_title(p)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
    return M,tracer,alignment


def affine_global_dp(seq_1,seq_2,S,g_open,g_ext,N=10,max_i=None,disp_scoring=False,disp_tracer=False,
              disp_alignments=False,high_low=False):
    
    #set max iterations
    if max_i is None: max_i = N**2
    
    #initialize the matrix
    m = len(seq_1); n = len(seq_2)
    M = np.zeros([m+1,n+1])
    M[0,1:] = g_open + g_ext*np.arange(0,n,1)
    M[1:,0] = g_open + g_ext*np.arange(0,m,1)
    L = np.copy(M); U = np.copy(M)
    L[1:,0] = L[1:,0]+g_open; U[0,1:] = U[0,1:]+g_open #avoiding Gotoh's error
    
    #fill up
    tracer = np.zeros([np.shape(M)[0],np.shape(M)[1],7])

    for i in range(1,m+1):
        for j in range(1,n+1):
            l_arr = np.array([M[i,j-1]+g_open,L[i,j-1]+g_ext])
            L[i,j] = np.max(l_arr)
            l_where = l_arr==np.max(l_arr)
                        
            u_arr = np.array([M[i-1,j]+g_open,U[i-1,j]+g_ext])
            U[i,j] = np.max(u_arr)
            u_where = u_arr==np.max(u_arr)
                        
            m_arr = np.array([M[i-1,j-1]+(S[seq_1[i-1]][seq_2[j-1]]),U[i,j],L[i,j]])
            M[i,j] = np.max(m_arr)
            m_where = m_arr==np.max(m_arr)
            
            idx = np.hstack([m_where,u_where,l_where])
            tracer[i,j,idx] = 1
    
    #traceback
    alignment = traceback_iterator(tracer,seq_1,seq_2,affine=True,high_low=high_low,N=N,max_i=max_i)
    
    if (disp_alignments is True):
        print("Max. score = "+str(max(M[-1,-1],L[-1,-1],U[-1,-1])))
        for i,e in enumerate(alignment):
            print(str(i+1)+".","\n",e[0],"\n",e[1],"\n")
    
    if disp_scoring is True:
        fig, ax = plt.subplots(1,3,figsize = (12,4))
        for i,p in enumerate([[M,"$M$"],[U,"$U$"],[L,"$L$"]]):
            sns.heatmap(p[0],linecolor='white',linewidth=1,square=True,cbar=True,cbar_kws={"shrink": .5},ax=ax[i]);
            ax[i].set_title(p[1])
        fig.suptitle("Scoring Matrices");
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if disp_tracer is True:
        fig, ax = plt.subplots(3,3,figsize = (6,6))

        for i,p in enumerate([[0,0,r"$\nwarrow_{M}$"],[0,1,r"$\odot_{U}$"],[0,2,r"$\odot_{L}$"],
                              [1,0,r"$\uparrow_{M}$"],[1,1,r"$\uparrow_{U}$"],
                              [2,0,r"$\leftarrow_{M}$"],[2,2,r"$\leftarrow_{L}$"]]):
            sns.heatmap(tracer[:,:,i],cmap="coolwarm",vmin=-0.5,vmax=0.6,linecolor='white',
                        linewidth=1,square=True,cbar=False,ax=ax[p[0]][p[1]])
            ax[p[0]][p[1]].set_title(p[2])

        fig.delaxes(ax[1][2]); fig.delaxes(ax[2][1])
        fig.suptitle("tracer sub-matrices")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]);
    
    return M,L,U,tracer,alignment


def local_dp(seq_1,seq_2,S,g,N=10,max_i=None,disp_scoring=False,disp_tracer=False,
              disp_alignments=False,high_low=False):
    
    #set max iterations
    if max_i is None: max_i = N**2
    
    #initialize the matrix
    m = len(seq_1); n = len(seq_2)
    M = np.zeros([m+1,n+1])
    
    #fill up
    tracer = np.zeros([np.shape(M)[0],np.shape(M)[1],3])
    for i in range(1,m+1):
        for j in range(1,n+1):
            arr = np.array([M[i-1,j-1]+(S[seq_1[i-1]][seq_2[j-1]]),(M[i-1,j]+g),(M[i,j-1]+g)])
            M[i,j] = np.max(np.hstack([arr,0]))
            idx = np.where(arr==np.max(arr))*int(M[i,j]!=0)
            tracer[i,j,idx] = 1
    
    #traceback
    alignment = traceback_iterator(tracer,seq_1,seq_2,mat=M,local=True,N=N,max_i=max_i,high_low=high_low)
    
    if disp_alignments is True:
        print("Max. score = "+str(np.max(M)))
        for i,e in enumerate(alignment):
            print(str(i+1)+".","\n",e[0],"\n",e[1],"\n")
    
    if disp_scoring is True:
        plt.figure(figsize = (5,5))
        sns.heatmap(M,linecolor='white',linewidth=1,cmap="viridis",square=True)
    
    if disp_tracer is True:
        fig, ax = plt.subplots(1,3,figsize = (12,4))
        for i,p in enumerate([r"$\nwarrow$",r"$\uparrow$",r"$\leftarrow$"]):
            sns.heatmap(tracer[:,:,i],linecolor='white',linewidth=1,
                        cmap="coolwarm",vmin=-0.5,vmax=0.6,square=True,cbar=False,ax=ax[i]);
            ax[i].set_title(p)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
    return M,tracer,alignment


def traceback(tracer,seq_1,seq_2,mat=None,local=False,affine=False,representation=False,roadmap=0):
    
    #get sequence lengths
    m = len(seq_1); n = len(seq_2)
    
    #convert to numpy arrays
    x = np.array(list(seq_1),dtype='object')
    y = np.array(list(seq_2),dtype='object')
    
    #set start location
    if local is False: st = [m+1,n+1]
    else:
        if roadmap == 0: r = np.random.choice(range(np.size(np.where(mat==np.max(mat))[0])),1)[0] #random maxima
        elif roadmap == 1: r = -1 #highroad
        elif roadmap == 2: r = 0 #lowroad
        st = [(np.where(mat==np.max(mat))[0][r])+1,(np.where(mat==np.max(mat))[1][r])+1]
        
        #set starting gaps based on the start location
        start_size = ((m-st[0])-(n-st[1])) #how many gaps and for which sequence
        start_gap = (['-']*abs(start_size))
        if start_size>0:
            y=np.append(y,start_gap)
        elif start_size<0:
            x=np.append(x,start_gap)

    if affine is True: 
        Tr = np.zeros([7]) #define a 7x1 Tr array
        st_lv = 0 #start in midgard
        
    while ((st[0]>1) & (st[1]>1)):
       
        B = np.zeros([2,2]) #define 2x2 box
        
        if representation is True: B = np.copy(tracer[st[0]-2:st[0],st[1]-2:st[1]])
        
        elif affine is False:
            B[0,0] = np.copy(tracer[st[0]-1,st[1]-1,0])
            B[0,1] = np.copy(tracer[st[0]-1,st[1]-1,1])
            B[1,0] = np.copy(tracer[st[0]-1,st[1]-1,2])
            B[1,1] = 1
    
        else:
            #tracer
            Tr[0] = np.copy(tracer[st[0]-1,st[1]-1,0])
            Tr[1] = np.copy(tracer[st[0]-1,st[1]-1,1])
            Tr[2] = np.copy(tracer[st[0]-1,st[1]-1,2])
            Tr[3] = np.copy(tracer[st[0]-1,st[1]-1,3])
            Tr[4] = np.copy(tracer[st[0]-1,st[1]-1,4])
            Tr[5] = np.copy(tracer[st[0]-1,st[1]-1,5])
            Tr[6] = np.copy(tracer[st[0]-1,st[1]-1,6])
        
        #bifurcations for non affine penalties
        if affine is False:
            check = np.array([B[1,0]==1,B[0,0]==1,B[0,1]==1])
            choose = np.array([[1,0],[0,0],[0,1]])
            if np.sum(check)>1:
                B[0,0] = 0; B[0,1] = 0; B[1,0] = 0 #reset
                if roadmap == 0: r = np.random.choice(np.where(check)[0],1)[0] #random turn
                elif roadmap == 1: r = np.where(check)[0][-1] #highroad
                elif roadmap == 2: r = np.where(check)[0][0] #lowroad
                else: raise Exception("roadmap only accepts 0: random turning, 1: highroad, 2: lowroad")
                B[choose[r][0],choose[r][1]] = 1
        
        #bifurcations for affine penalties
        else:    
            levels = [[0,3],[3,5],[5,7]]
            for l in levels:
                if np.sum(Tr[l[0]:l[1]])>1:
                    choose = np.where(Tr[l[0]:l[1]]==1)[0]
                    Tr[l[0]:l[1]] = 0
                    if roadmap == 0: r = np.random.choice(choose,1)[0] #random turning
                    elif roadmap == 1: r = choose[-1] #highroad
                    elif roadmap == 2: r = choose[0] #lowroad
                    else: raise Exception("roadmap only accepts 0: random turning, 1: highroad, 2: lowroad")
                    Tr[l[0]:l[1]][r] = 1
                  
            #level up-down
            if ((Tr[0]==1) & (st_lv==0)): #diagonal
                B[0,0] = 1

            if ((Tr[1]==1) & (st_lv==0)): #level up
                st_lv = 1

            if ((Tr[2]==1) & (st_lv==0)): #level down
                st_lv = 2

            if ((Tr[4]==1) & (st_lv==1)): #move up
                B[0,1] = 1
                
            if ((Tr[3]==1) & (st_lv==1)): #move up back to main
                st_lv = 0
                B[0,1] = 1
                
            if ((Tr[6]==1) & (st_lv==2)): #move left
                B[1,0] = 1

            if ((Tr[5]==1) & (st_lv==2)): #move left back to main
                st_lv = 0
                B[1,0] = 1
        
        if local is True:
            if (mat[st[0]-1,st[1]-1]==0):
                break
                
        #movements
        if B[0,1]==1: #upward
            y = np.insert(y,st[1]-1,'-') #add a gap
            st[0] = st[0]-1
            
        if B[1,0]==1: #leftward
            x = np.insert(x,st[0]-1,'-') #add a gap
            st[1] = st[1]-1

        if B[0,0]==1: #diagonal
            st[1] = st[1]-1
            st[0] = st[0]-1
        
    #some end gaps are left when you hit the upper/lower end of the matrix or a 0
    end_size = (np.size(x)-np.size(y)) #how many gaps and for which sequence
    end_gap = (['-']*abs(end_size))
    if end_size>0:
        y=np.insert(y,0,end_gap)
    elif end_size<0:
        x=np.insert(x,0,end_gap)

    #check no overlapping gaps
    x = np.where(((x=='-')&(y=='-')),None,x)
    y = np.where((x==None),'',y)
    x = np.where((x==None),'',x)

    return np.sum(x),np.sum(y)


def traceback_iterator(tracer,seq_1,seq_2,mat=None,N=10,max_i=100,high_low=False,
                       affine=False,local=False,representation=False):
    alignment = []
    if high_low is False:
        i = 0; s=0
        while ((s<N) & (i<max_i)):
            alignment.append(traceback(tracer,seq_1,seq_2,mat=mat,affine=affine,
                                       local=local,representation=representation,roadmap=0))        
            s = len(list(set(map(tuple,alignment))))
            i+=1 
            if ((s<N)&(i>=max_i)): 
                print("WARNING: "+str(max_i)+" iterations exceeded;"+" <"+str(N)
                      +" alignments found: to continue searching, increase max_i.")
    else:
        for i in range(1,3,1):
            alignment.append(traceback(tracer,seq_1,seq_2,mat=mat,affine=affine,local=local,
                                       representation=representation,roadmap=i))
    return list(set(map(tuple,alignment)))
