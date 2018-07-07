from __future__ import print_function
import ROOT, sys
from ROOT import TChain
from larcv import larcv
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN
from larcv import larcv

# A utility function to compute the 2D (X,Y) range to zoom-in so that it avoids showing zero region of an image.
def get_view_range(image2d):
    nz_pixels=np.where(image2d>0.0)
    ylim = (np.min(nz_pixels[0])-5,np.max(nz_pixels[0])+5)
    xlim = (np.min(nz_pixels[1])-5,np.max(nz_pixels[1])+5)
    # Adjust for allowed image range
    ylim = (np.max((ylim[0],0)), np.min((ylim[1],image2d.shape[1]-1)))
    xlim = (np.max((xlim[0],0)), np.min((xlim[1],image2d.shape[0]-1)))
    return (xlim,ylim)

def show_event(entry=-1, plane=0):
    # Create TChain for data image
    chain_image2d = ROOT.TChain('image2d_data_tree')
    chain_image2d.AddFile('data/test_10k.root')
    # Create TChain for label image
    chain_label2d = ROOT.TChain('image2d_segment_tree')
    chain_label2d.AddFile('data/test_10k.root')
    
    if entry < 0:
        entry = np.random.randint(0,chain_label2d.GetEntries())

    chain_label2d.GetEntry(entry)
    chain_image2d.GetEntry(entry)

    # Let's grab a specific projection (1st one)
    image2d = larcv.as_ndarray(chain_image2d.image2d_data_branch.as_vector()[plane])
    label2d = larcv.as_ndarray(chain_label2d.image2d_segment_branch.as_vector()[plane])

    # Get image range to focus
    #xlim, ylim = get_view_range(image2d)
    
    # Dump images
    #fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(18,12), facecolor='w')
    #ax0.imshow(image2d, interpolation='none', cmap='jet', origin='lower')
    #ax1.imshow(label2d, interpolation='none', cmap='jet', origin='lower',vmin=0., vmax=3.1)
    #ax0.set_title('Data',fontsize=20,fontname='Georgia',fontweight='bold')
    #ax0.set_xlim(xlim)
    #ax0.set_ylim(ylim)
    #ax1.set_title('Label',fontsize=20,fontname='Georgia',fontweight='bold')
    #ax1.set_xlim(xlim)
    #ax1.set_ylim(ylim)
    
    return (np.array(image2d), np.array(label2d))

#the lines commented out should be included to see the scatter plot of the shower produced
def plot_best_fit(image_array):
    weights = np.array(image_array)
    x = np.where(weights>0)[1]
    y = np.where(weights>0)[0]
    #plt.plot(x, y,'.')
    size = len(image_array) * len(image_array[0])

    y = np.zeros((len(image_array), len(image_array[0])))
    for i in range(len(np.where(weights>0)[0])):
        y[np.where(weights>0)[0][i]][np.where(weights>0)[1][i]] = np.where(weights>0)[0][i]
    y = y.reshape(size)
    x = np.array(range(len(image_array)) * len(image_array[0]))
    weights = weights.reshape((size))
    b, m = polyfit(x, y, 1, w=weights)
    angle = math.atan(m) * 180/math.pi
    #plt.plot(x, b+m*x, '-')
    return b,m, angle
# Let's look at one specific event entry

def energy(event, plane=0):
    image2d, label2d = show_event(event, plane)
    unique_values, unique_counts = np.unique(label2d, return_counts=True)
    #print('Label values:',unique_values)
    #print('Label counts:',unique_counts)
    categories = ['Background','Shower','Track']
    fig, axes = plt.subplots(1, len(unique_values), figsize=(18,12), facecolor='w')
    xlim,ylim = get_view_range(image2d)

    for index, value in enumerate(unique_values):
        ax = axes[index]
        mask = (label2d == value)
        Image = image2d*mask
        ax.imshow(Image, interpolation='none', cmap='jet', origin='lower')
        #print("Event", categories[index])
        #print("Energy: {0:.2f} MeV".format(np.sum((Image)/100)))
        ax.set_title(categories[index],fontsize=20,fontname='Georgia',fontweight='bold')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if index==1:
            showerImage = np.array(Image)
            b, m, angle=plot_best_fit(showerImage)
            ax.set_title("{0} {1:.2f} deg".format(categories[index], angle),fontsize=20,fontname='Georgia',fontweight='bold')
            print("Angle of the shower: {0:.2f}".format(angle))
            #print("Energy of the shower: {0:.2f} MeV".format(np.sum((Image)/100)))
            shower_energy = np.sum(Image/100)
            x = np.array(range(0,len(showerImage[0])))
            y = x*m + b
            ax.plot(x, y)
    plt.show()

    particle_mcst_chain = TChain("particle_mcst_tree")
    particle_mcst_chain.AddFile("data/test_10k.root")
    particle_mcst_chain.GetEntry(event)
    cpp_object = particle_mcst_chain.particle_mcst_branch

    #print('particle_mcst_tree contents:')
    energy_deposit = 0
    for particle in cpp_object.as_vector():
        if abs(particle.pdg_code()) == 22 or abs(particle.pdg_code())==11:
            energy_deposit += particle.energy_deposit()
    #print("Energy Deposit: ", energy_deposit)
    return energy_deposit, shower_energy

def singleParticle(event):
    particle_mcst_chain = TChain("particle_mcst_tree")
    particle_mcst_chain.AddFile("data/test_10k.root")
    particle_mcst_chain.GetEntry(event)
    cpp_object = particle_mcst_chain.particle_mcst_branch
    particles = []
    single = True
    showerParticles = [11, -11, 111, 22]
    for particle in cpp_object.as_vector():
        particles.append(particle.pdg_code())
    for sParticle in showerParticles:
        if particles.count(sParticle) > 1:
            single=False
            return False, 0
    sumCount = 0
    trackIDs = []

    for particle in particles:
        if particle==22 or particle==11 or particle==-11 or particle==111:
            sumCount += 1
    for particle in cpp_object.as_vector():
        if abs(particle.pdg_code()) == 22 or abs(particle.pdg_code()) == 111 or abs(particle.pdg_code()) == 11:
            trackIDs.append(particle.track_id())
    singleEvent = single and (sumCount == 1)

    return singleEvent, trackIDs

def momentum(event, plane, plotLine):
    singleTruth, trackIDs = singleParticle(event)
    if singleTruth:
        image2d, label2d = show_event(event, plane)
        unique_values, unique_counts = np.unique(label2d, return_counts=True)
        categories = ['Background','Shower','Track']
        if plotLine:
            fig, axes = plt.subplots(1, len(unique_values), figsize=(18,12), facecolor='w')
            xlim,ylim = get_view_range(image2d)
        for index, value in enumerate(unique_values):
            mask = (label2d == value)
            Image = image2d*mask
            if plotLine:
                ax = axes[index]
                ax.imshow(Image, interpolation='none', cmap='jet', origin='lower')
                ax.set_title(categories[index],fontsize=20,fontname='Georgia',fontweight='bold')
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            if index==1:
                showerImage = np.array(Image)
                b, m, angle=plot_best_fit(showerImage)
                print("Angle of the shower: {0:.2f}".format(angle))
                x = np.array(range(0,len(showerImage[0])))
                y = x*m + b
                if plotLine:
                    ax.set_title("{0} {1:.2f} deg".format(categories[index], angle),fontsize=20,fontname='Georgia', fontweight='bold')
                    ax.plot(x, y)
        if plotLine: plt.show()
        print("Plot is", plotLine)

        particle_mcst_chain = TChain("particle_mcst_tree")
        particle_mcst_chain.AddFile("data/test_10k.root")
        particle_mcst_chain.GetEntry(event)
        cpp_object = particle_mcst_chain.particle_mcst_branch

        for particle in cpp_object.as_vector():
            if plotLine:
                print(particle.dump())
            if particle.track_id() in trackIDs:
                if plane==0:
                    trueAngle = math.degrees(math.atan(particle.py()/particle.px()))
                elif plane==1:
                    trueAngle = math.degrees(math.atan(particle.pz()/particle.py()))
                else:
                    trueAngle = math.degrees(math.atan(particle.pz()/particle.px()))
        print("True angle of the shower: {0:.2f}".format(trueAngle))
    else:
        return False

    return angle, trueAngle, event

def find_cluster(event, plane):
    image2d, label2d = show_event(event, plane)
    unique_values, unique_counts = np.unique(label2d, return_counts=True)
    categories = ['Background','Shower','Track']
    if len(unique_values) != 3: return # some events don't have a track/shower :/
    for index, value in enumerate(unique_values):
        mask = (label2d == value)
        Image = image2d*mask
        if index==1:
            showerImage = np.array(Image)
        elif index==2:
            trackImage = np.array(Image)

    scatterX = np.where(showerImage>0)[1]
    scatterY = np.where(showerImage>0)[0]
    trackX = np.where(trackImage>0)[1]
    trackY = np.where(trackImage>0)[0]
##############################################################
    track = []
    for i,j in zip(trackX, trackY):
        track.append([i,j])
    model_t = DBSCAN(eps=8, min_samples=14).fit(track)
    num_labels_t = np.unique(np.array(model_t.labels_))
    t_labels_count = len(num_labels_t)
###############################################################
    X = []
    for i,j in zip(scatterX, scatterY):
        X.append([i, j])
    model = DBSCAN(eps=8, min_samples=14).fit(X)
    num_labels = np.unique(np.array(model.labels_))
    nimages = len(num_labels) + 1
    print("Number of images ", nimages)
    if nimages%3==0:
        nrows = 3
        ncols = nimages/3
    elif nimages%2==0:
        nrows = 2
        ncols = nimages/2
    else:
        nrows = 3
        ncols = nimages/3 + 1
    plt.subplot(nrows, ncols, 1)
    plt.plot(scatterX,scatterY,".",c="blue")
    plt.plot(trackX,trackY,".",c="red")
    plt.title("Original Event (Track: red; Shower: blue)")
###############################################################
    cluster_labels = {}
###############################################################
    for counter, cluster in enumerate(num_labels):
        x_0 = []
        y_0 = []
        x = []
        y = []
        for index, i in enumerate(model.labels_):
            if i==cluster: x_0.append(index)

        for index, i in enumerate(X):
            if index in x_0: y_0.append(X[index])

        for i in y_0:
            x.append(i[0])
            y.append(i[1])
        plt.subplot(nrows, ncols, counter+2)
################################################################
        C = []
        for i,j in zip(x, y):
            C.append([i, j])
        model_c = DBSCAN(eps=8, min_samples=14).fit(C)
        num_labels_c = np.unique(np.array(model_c.labels_))
        c_labels_count = len(num_labels_c)

        T = []
        for i,j in zip(x, y):
            T.append([i,j])
        for k,l in zip(trackX, trackY):
            T.append([k,l])
        model_tc = DBSCAN(eps=8, min_samples=14).fit(T)
        num_labels_tc = np.unique(np.array(model_tc.labels_))
        tc_labels_count = len(num_labels_tc)
        cluster_labels[cluster] = (t_labels_count, c_labels_count, tc_labels_count)
################################################################
        plt.plot(x,y,".")

        if cluster==-1:
            plt.title("Background Events")
        else:
            plt.title("Cluster " + str(cluster))
################################################################
    print("Only track labels        Only cluster labels           Track & Cluster labels")
    print(cluster_labels)

    count_tc = 0
    clusters = []
    cluster_count = []
    for cluster_num, label in zip(cluster_labels.keys(), cluster_labels.values()):
        if label[2] == 1:
            count_tc += 1
            clusters.append(cluster_num)
    if count_tc == 1:
        print("Cluster " + str(clusters[0]))
    elif count_tc > 1:
        for counter, cluster in enumerate(clusters):
            x_0 = []
            y_0 = []
            x = []
            y = []
            for index, i in enumerate(model.labels_):
                if i==cluster: x_0.append(index)

            for index, i in enumerate(X):
                if index in x_0: y_0.append(X[index])

            for i in y_0:
                x.append(i[0])
                y.append(i[1])
            cluster_count.append(len(x))
        cluster_index = cluster_count.index(max(cluster_count))
        print("Cluster " + str(clusters[cluster_index]))
        #take the one with the most samples
    else:
        #min of the difference between tc_label and c_label
        #if more than one:
        #    take the one with most events
        cluster_diff = {}
        for cluster_num, label in zip(cluster_labels.keys(), cluster_labels.values()):
            cluster_diff[cluster_num] = label[2] - label[1]
        min_diff = min(cluster_diff.values())
        if cluster_diff.values().count(min_diff) == 1:
            for label, counts in cluster_diff.iteritems():
                if counts == min_diff:
                    print("Cluster ", label)
        else:
            min_clusters = []
            for label, counts in cluster_diff.iteritems():
                if counts == min_diff:
                    min_clusters.append(label)
            min_counts = []
            for counter, cluster in enumerate(min_clusters):
                x_0 = []
                y_0 = []
                x = []
                y = []
                for index, i in enumerate(model.labels_):
                    if i==cluster: x_0.append(index)

                for index, i in enumerate(X):
                    if index in x_0: y_0.append(X[index])

                for i in y_0:
                    x.append(i[0])
                    y.append(i[1])
                min_counts.append(len(x))
            cluster_index = min_counts.index(max(min_counts))
            print("Cluster " + str(min_clusters[cluster_index]))

################################################################
    plt.show()

if __name__ == "__main__":
    #true = []
    #calculated = []
    #events = []
    for i in range(int(sys.argv[1]),int(sys.argv[2])):
         find_cluster(i, 0)
         #if singleParticle(i)[0]:
             #angle, trueAngle, event = momentum(i, 1, False)
             #true.append(trueAngle)
             #calculated.append(angle)
             #events.append(event)
    #    deposit, shower = energy(i)
    #    if deposit>0:
    #        true.append(deposit)
    #        calculated.append(shower)

    #percent_array = []
    #for i in range(len(true)):
        #percent_error = abs((abs(true[i]) - abs(calculated[i])))/abs(true[i]) * 100
        #if percent_error < 100:
            #percent_array.append(percent_error)
        #else:
            #percent_array.append(105)
            #myEvent = events[i]
            #momentum(myEvent, 1, True)
    #plt.hist(percent_array, bins=30)
    #plt.title("Error in Angle Prediction")
    #plt.xlabel("Percentage Error")
    #plt.ylabel("Frequency")
    #plt.show()
