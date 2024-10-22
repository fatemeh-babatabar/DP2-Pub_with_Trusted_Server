import itertools
import pandas as pd
from Server.PrivBayes import PrivBayes
from Server.Clustering import clustering 
from Server.PRAM import PRAM
from Server.StatisticalAnalyses.TVD import TVD 


def read_dataset():
        input_domain='Data4-coarse.domain'
        input_data='Data4-coarse.dat'
    
        fd=open(input_domain,'r')
        head_line=fd.readline()
        readrow=head_line.split(" ")
        att_num=int(readrow[0])
        fd.close()
    
        ##########################################################################################
        #Get the attributes info in the domain file
        multilist = []
        fp=open(input_data,"r")
        fp.readline();  ###just for skip the header line
        
    
        while 1:
            line = fp.readline()
            if not line:
                break 
            
            line=line.strip("\n")
            temp=line.split(",")
            temp2 = [str(i)+'a' for i in temp]
            multilist.append(temp2)
        fp.close()

        colomn = [str(i)+'a' for i in range(att_num)]
        df = pd.DataFrame(multilist, columns = colomn)
        df.to_csv("dataset.csv", index=False)
        return df

def attributes_domain():
        input_domain='Data4-coarse.domain'
        fd=open(input_domain,'r')
        head_line=fd.readline()
        readrow=head_line.split(" ")
    
        domain = dict()
        i=0
        while 1:
            line = fd.readline()
            if not line:
                break
        
            line=line.strip("\n")
            readrow=line.split(" ")
            start_x=0
            dom = []
            for eachit in readrow:
                start_x=start_x+1
                eachit.rstrip()
                if start_x>3:
                    dom.append(str(eachit)+'a')
            domain.update({str(i)+'a':dom})
            i=i+1
        fd.close()
        return domain

if __name__ == "__main__":
        dataset = read_dataset()
        domains = attributes_domain()

        client_dataset = dataset.copy(deep=True)
        client_dataset2 = dataset.copy(deep=True)

        epsilon = 1
        e1 = e2 = epsilon/2

        # constructing bayesian network  (k = 2)
        privBayes = PrivBayes(client_dataset, e1)

        # constructing clusters of size 3 
        len3 = False
        while not len3:
            clu = clustering(privBayes.BN, domains)
            len3 = all(len(sublist) < 4 for sublist in clu.clusters)

        # invariant PRAM
        P = PRAM(e2, clu.clusters, clu.PBC, domains, client_dataset2)

        _tvd = 0
        # NLTCS dataset..... α = 3
        # constructing all subset with size 3
        subsets_of_size_3 = list(itertools.combinations(domains.keys(), 3))
        for subset in subsets_of_size_3:
            _tvd += TVD(dataset[list(subset)], P.randomized_data[list(subset)] , domains).tvd

        _tvd_average = _tvd / len(subsets_of_size_3)

        print("average total variation distance : "+ str(_tvd_average))