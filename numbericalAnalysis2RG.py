import numpy as np
import math
import time



time_kol = 0

num_list = [4,8,16,32,64,128,256,512,1024]

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
            x[i] = temp[i]
            
        result = math.sqrt(result)

        if abs(result)<=0.0001:
            print(result)
            break
        else:
            bar+=1
    time_of_program = time.time() - start_time
    time_kol += time_of_program

    
    x1 = np.linalg.solve(a,b)      
    
    print("Solution:")
    print(x)
    print("solution with np.linalg.solve")
    print(x1)
    print("Error:")
    print(result)
    print("x - x(with np.linalg:")
    print(x - x1)
    print("tedad:")
    print(bar+1)
    
    print("********************************************************")
    print("--- %s seconds ---" %(time_of_program))

    
print("*****************************************************")
print("--- %s seconds ---" %(time_kol))