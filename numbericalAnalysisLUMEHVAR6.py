import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches

def zarb(A, B) :
    temp1 =0
    C = []
    n = len(A)
    for i in range(0, n) :
        row = []        
        for j in range(0, n) :
            result = 0
            for k in range(0, n) :
                result += A[i][k] * B[k][j]
                temp1+=1
            row.append(result)
        C.append(row)
    return C,temp1


def mehvargiri(a,n):

    p=np.diag([ 1.0 for i in range(n)],0)
    temp2 = 0

    for i in range(0, n):

        maxd = abs(a[i][i])
        maxRow = i
        for j in range(i + 1, n):
            temp2+=1
            if abs(a[j][i]) > maxd:
                maxd = abs(a[j][i])
                maxRow = j
        p[i],p[maxRow] = p[maxRow] , p[i]
    C,temp1=zarb(p,a)
    return C,(temp1+temp2)


def gauss(matrix,a,lower,n):
    temp = 0

    for j in range(n):
        lower[j][j] =1
        for i in range(j+1):
            sum1 = 0

            for k in range(i):
                temp +=1
                sum1 += a[k][j]*lower[i][k]
            
            a[i][j] = matrix[i][j] - sum1

        for i in range(j+1, n):
            sum2 = 0
            for k in range(j):
                temp +=1
                sum2 += a[k][j]*lower[i][k]
           
            lower[i][j] = (matrix[i][j] - sum2) / a[j][j]

    return lower,a,temp
 
    



all_times = []
num_list = [4,8,16,32,64,128,512,1024]
all_temps = []
for n in num_list:
    
    s = (n,n)
    lower = np.diag([ 1.0 for i in range(n)],0)
    upper = np.zeros(s)
    b = [1.0 for i in range(n)]
    
    
    a = np.zeros(s)
   
    #A = [[0 for j in range(n + 1)] for i in range(n)]
    """
    for i in range(n):
        b.append(int(input("Element:")))
    """
    # A matrix
    for i in range(n):
        
        for j in range(n):
            if i==j:
                a[i][j]=4
            elif i==j+1 or j==i+1 :
                a[i][j]=-1
                

    print(a)
    

    # Print input


    # Calculate solution
    start_time = time.time()
    a,temp1 = mehvargiri(a,n)
    upper,lower,temp = gauss(a,upper,lower,n)
    all_temps.append(temp1+temp)
    time_of_program = time.time() - start_time
    all_times.append(time_of_program)


    for i in range(n):
 
        # Lower
        for j in range(n):
            print('%2f'%lower[i][j], end="\t")
        print("", end="\t")
 
        # Upper
        for j in range(n):
            print('%2f'%upper[i][j], end="\t")
        print("")

print("**************************************************************")
for i in range(len(num_list)):
    print("n:",num_list[i])
    print("time:",all_times[i])
    print("")
print("****************************************************************")

print("**************************************************************")
for i in range(len(num_list)-1):
    print("n:%d , n+1:%d"%(num_list[i+1],num_list[i]))
    print("nesbat:",all_times[i+1]/all_times[i])
    print("nasbat ha 8 ast.pas 2^3 = 8 pas O(n3) ast.")
print("****************************************************************")


print("**************************************************************")
for i in range(len(num_list)):
    print("n:",num_list[i])
    print("temps:",all_temps[i])
    print("")
print("****************************************************************")

print("**************************************************************")
for i in range(len(num_list)-1):
    print("n:%d , n+1:%d"%(num_list[i+1],num_list[i]))
    print("nesbat:",all_temps[i+1]/all_temps[i])
    print("nasbat ha 8 ast.pas 2^3 = 8 pas O(n3) ast.")
print("****************************************************************")


xpoints = np.array(num_list)
ypoints = np.array(all_temps)
y1points =np.array(all_times)

allDegree = np.polyfit(xpoints, ypoints, len(num_list))
print(allDegree)
allDegrees1 = np.polyfit(xpoints, y1points, len(num_list))
print(allDegrees1)


temp = 0
for i in range(len(allDegree)):
    if abs(allDegree[i])<0.001:
        allDegree[i] = 0
    else:
        temp +=1

print(temp-1)
print("zarayeb:")
for i in range(len(allDegree)):
    if allDegree[i]!= 0:
        print(allDegree[i])

plt.xlabel('x - axis')
# Set the y axis label of the current axis.
plt.ylabel('y - axis')


aLabel = plt.plot(xpoints,ypoints)
plt.plot(xpoints, ypoints, 'o')

bLabel = plt.plot(xpoints,y1points)
plt.plot(xpoints, y1points,'g^')


#plt.legend([aLabel , bLabel] , ["tedad","times"])
plt.show()


