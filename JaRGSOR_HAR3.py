import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.style.use('classic')

time_kol1 = 0
time_kol2 = 0
time_kol3 = 0
all_times1 =[]
all_times2 =[]
all_times3 = []
deghat1 = []
deghat2 = []
deghat3 = []

num_list = [4,8,16,32,64,128,512,1024]


w = float(input("w:"))
for n in num_list:


    bar = 0
    s=(n,n)
    result = 0
    b = [1 for i in range(n)]

    #x = np.zeros(n)
    
    x = []
    print("bare ye "+str(n)+" matrix")
    for i in range(n):
        x.append(float(input("Element:")))
    
    x = np.array(x)
    
    print(x)
    
    a = np.zeros(s)

    for i in range(n):
        
        for j in range(n):
            if i==j:
                a[i][j]=4
            elif i==j+1 or j==i+1 :
                a[i][j]=-1


    
    start_time = time.time()
    
    while True:
        temp=[0 for i in range(n)]
        result = 0
        for i in range(n):
            r1=0
            for j in range(i):
                r1 += a[i][j]*x[j]
            for j in range(i+1,n):
                r1 +=a[i][j]*x[j]
            
            temp[i] = (b[i] - r1)/a[i][i]
            result += ((temp[i]-x[i])*(temp[i]-x[i]))
        result = math.sqrt(result)
        x = temp
        if abs(result)<=0.0001:
            print(result)
            break
        else:
            bar+=1    


    time_of_program1 = time.time() - start_time
    time_kol1 += time_of_program1
    all_times1.append(time_of_program1)
    
    x1 = np.linalg.solve(a,b)      
    print("JACOBY")
    print("Solution:")
    print(x)
    print("solution with np.linalg.solve")
    print(x1)
    print("Error:")
    print(result)
    print("x - x(with np.linalg):")
    deghat = x-x1
    print(deghat)
    deghat1.append(deghat)
    print("tedad:")
    print(bar+1)
    
    print("********************************************************")
    print("--- %s seconds ---" %(time_of_program1))
    print("********************************************************")
    
    
    
    #RG


    while True:
        temp=[0 for i in range(n)]
        result = 0
        for i in range(n):
            r1=0
            for j in range(i):
                r1 += a[i][j]*x[j]
            for j in range(i+1,n):
                r1 +=a[i][j]*x[j]
            
            temp[i] = (b[i] - r1)/a[i][i]
            
            result += ((temp[i]-x[i])*(temp[i]-x[i]))
            x[i] = temp[i]
        result = math.sqrt(result)

        if abs(result)<=0.0001:
            print(result)
            break
        else:
            bar+=1


    time_of_program2 = time.time() - start_time
    time_kol2 += time_of_program2
    all_times2.append(time_of_program2)
    

    x1 = np.linalg.solve(a,b)      
    print("Guass_sidel")
    print("Solution:")
    print(x)
    print("solution with np.linalg.solve")
    print(x1)
    print("Error:")
    print(result)
    print("x - x(with np.linalg:")
    deghat = x-x1
    print(deghat)
    deghat2.append(deghat)
    print("tedad:")
    print(bar+1)
    
    print("********************************************************")
    print("--- %s seconds ---" %(time_of_program2))



    #SOR
    start_time = time.time()
    while True:
        temp=[0 for i in range(n)]
        result = 0
        for i in range(n):
            r1=0
            for j in range(i):
                r1 += a[i][j]*x[j]
            for j in range(i+1,n):
                r1 += a[i][j]*x[j]
            #if i==j:continue

            temp[i] = (w*(b[i] - r1))/a[i][i]
            temp[i] += (1-w)*x[i]
            
            result += ((temp[i]-x[i])*(temp[i]-x[i]))
            
            x[i] = temp[i]
        
        result = math.sqrt(result)


        if result<=0.0001:
            print(result)
            break
        else:
            bar+=1

    time_of_program3 = time.time() - start_time
    time_kol3 += time_of_program3
    all_times3.append(time_of_program3)


    x1 = np.linalg.solve(a,b)      
    
    print("Solution:")
    print(x)
    print("solution with np.linalg.solve")
    print(x1)
    print("Error:")
    print(result)
    print("x - x(with np.linalg:")
    deghat = x-x1
    print(deghat)
    deghat3.append(deghat)
    print("teadad")
    print(bar+1)

    print("********************************************************")
    print("--- %s seconds ---" %(time_of_program3))



for i in range(len(num_list)):
    print("n:",num_list[i])
    print("Jacobi")
    print("time:",all_times1[i])
    print("deghat:",deghat1[i])
    print("GS")
    print("time:",all_times2[i])
    print("deghat:",deghat2[i])
    print("time:",all_times3[i])
    print("deghat:",deghat3[i])
    print("with w :",w)

xpoints = np.array(num_list)
yTimes1points = np.array(all_times1)
yTimes2points = np.array(all_times2)
yTimes3points = np.array(all_times3)

plt.plot(xpoints, yTimes1points,"red")
plt.plot(xpoints, yTimes2points,"darkgreen")
plt.plot(xpoints, yTimes3points,"purple")

plt.plot(xpoints, yTimes1points,'o')
plt.plot(xpoints, yTimes2points,'o')
plt.plot(xpoints, yTimes3points,'o')
#'g^'

"""


ydeght1points = np.array(deghat1)
ydeght2points = np.array(deghat2)
ydeght3points = np.array(deghat3)


plt.plot(xpoints, ydeght1points ,"salmon")
plt.plot(xpoints, ydeght2points ,"lime")
plt.plot(xpoints, ydeght3points ,"pink")


plt.plot(xpoints, ydeght1points,'o')
plt.plot(xpoints, ydeght2points,'o')
plt.plot(xpoints, ydeght3points,'o')



"""

plt.show()
