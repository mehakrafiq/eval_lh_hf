from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import cm

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def compute_binary_labels(data, label0, label1):
    len0 = data[(data.section==label0) & (data.fs>=1)]
    len1 = data[(data.section==label1) & (data.fs>=1)]
    y0 = np.zeros(len(len0))
    y1 = np.ones(len(len1))
    y = np.concatenate((y0,y1))
    
    return y

def compute_features(data,x,y, label0, label1):
    x0 = data[x][(data.section==label0) & (data.fs>=1)] 
    x1 = data[x][(data.section==label1) & (data.fs>=1)] 
    y0 = data[y][(data.section==label0) & (data.fs>=1)] 
    y1 = data[y][(data.section==label1) & (data.fs>=1)] 
    x =  np.concatenate((x0,x1))
    y =  np.concatenate((y0,y1))
    x_avg =  compute_mean_feat(x)
    y_avg =  compute_mean_feat(y)
    feat =  np.vstack((x_avg,y_avg)).T
    return feat
    
def compute_features_general(data, x0, x1, y0,y1):
    xL = data[x0][(data.fs>=1)] 
    xH = data[x1][ (data.fs>=1)] 
    yL = data[y0][(data.fs>=1)] 
    yF = data[y1][(data.fs>=1)] 
    x =  np.concatenate((xL,xH))
    y =  np.concatenate((yL,yH))
    x_avg =  compute_mean_feat(x)
    y_avg =  compute_mean_feat(y)
    feat =  np.vstack((x_avg,y_avg)).T
    return feat

def compute_features_4D(x0,x1, df1, df2,df3, df4):
    cat1L = df1[x0][(df1.fs>=1)] 
    cat1H = df1[x1][ (df1.fs>=1)] 
    cat2L = df2[x0][(df2.fs>=1)] 
    cat2H = df2[x1][(df2.fs>=1)] 
    cat3L = df3[x0][(df3.fs>=1)] 
    cat3H = df3[x1][(df3.fs>=1)] 
    cat4L = df4[x0][(df4.fs>=1)] 
    cat4H = df4[x1][(df4.fs>=1)] 
    x =  np.concatenate((cat1L,cat2L,cat3L,cat4L))
    y =  np.concatenate((cat1H,cat2H,cat3H,cat4H))
    x_avg =  compute_mean_feat(x)
    y_avg =  compute_mean_feat(y)
    feat =  np.vstack((x_avg,y_avg)).T
    return feat

def compute_labels_general_4D(df1, df2,df3,df4, label1, label2,label3,label4):
    len0 = df1[ (df1.fs>=1)]
    len1 = df2[(df2.fs>=1)] 
    len2 = df3[(df3.fs>=1)]
    len3 = df4[(df4.fs>=1)]
    y0 = label1*np.ones(len(len0))
    y1 = label2*np.ones(len(len1))
    y2 = label3*np.ones(len(len2))
    y3 = label4*np.ones(len(len3))

    y = np.concatenate((y0,y1,y2,y3))
    
    return y
    

def compute_features_2D(x0,x1, df1, df2):
    xL = df1[x0][(df1.fs>=1)] 
    xH = df1[x1][ (df1.fs>=1)] 
    yL = df2[x0][(df2.fs>=1)] 
    yH = df2[x1][(df2.fs>=1)] 
    x =  np.concatenate((xL,yL))
    y =  np.concatenate((xH,yH))
    x_avg =  compute_mean_feat(x)
    y_avg =  compute_mean_feat(y)
    feat =  np.vstack((x_avg,y_avg)).T
    return feat

def compute_labels_general(df1, df2, label1, label2):
    len0 = df1[ (df1.fs>=1)]
    len1 = df2[(df2.fs>=1)]
    y0 = label1*np.ones(len(len0))
    y1 = label2*np.ones(len(len1))
    y = np.concatenate((y0,y1))
    
    return y



def combine_arrays(arrays):
    arr = np.array([])
    for i in arrays:
        arr =  np.concatenate((arr,i))
    return arr


def create_label(label_val, len_labels):
    arr = label_val*np.ones((len_labels))
    return arr

    
    
def compute_mean_feat(arr):
    vals = []
    for i in arr:
        vals.append(np.mean(i))
    return np.array(vals)
    

def plot_window_comparisons(feat1, feat2, feat3, model1, model2, model3, labels, var1, var2, flag, save_fig):
    
    if flag:

        fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3,figsize=(12, 4), tight_layout=True)
    
        X0, X1 = feat1[:, 0], feat1[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        plot_contours(ax1, model1, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax1.scatter(X0, X1, c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax1.set_ylabel('LF(ia)')
        ax1.set_xlabel('HF(ia)')
        ax1.set_xticks(())
        ax1.set_yticks(())
        ax1.set_title('SVM for '+ var1+ ' vs '+ var2 + ' (W=10)')

        X0, X1 = feat2[:, 0], feat2[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        plot_contours(ax2, model2, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax2.scatter(X0, X1, c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax2.set_ylabel('LF(ia)')
        ax2.set_xlabel('HF(ia)')
        ax2.set_xticks(())
        ax2.set_yticks(())
        ax2.set_title('SVM for '+ var1+ ' vs '+ var2 + ' (W=50)')

        X0, X1 = feat3[:, 0], feat3[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        plot_contours(ax3, model3, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax3.scatter(X0, X1, c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax3.set_ylabel('LF(ia)')
        ax3.set_xlabel('HF(ia)')
        ax3.set_xticks(())
        ax3.set_yticks(())
        ax3.set_title('SVM for '+ var1+ ' vs '+ var2 + ' (W=100)')
        if save_fig:
            plt.savefig('svm'+var1+'_'+var2+'.jpg')

        plt.show()


def plot_window(feat1, model1, labels, plot_fig, save_fig,filename=None):

        if plot_fig:
            fig, (ax1) = plt.subplots(nrows=1, ncols=1,figsize=(12, 12))
    
            X0, X1 = feat1[:, 0], feat1[:, 1]
            xx, yy = make_meshgrid(X0, X1)
            plot_contours(ax1, model1, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
            ax1.scatter(X0, X1, c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            ax1.set_ylabel('LF(ia)')
            ax1.set_xlabel('HF(ia)')
       
            ax1.set_title('SVM for '+ ' (W=10)')
            if save_fig:
                plt.savefig(filename+'.jpg')





def quadrant_chart(x, y, xtick_labels=None, ytick_labels=None, ax=None):

    # make the data easier to work with by putting it in a dataframe
    data = pd.DataFrame({'x': x, 'y': y})

    # let the user specify their own axes
    ax = ax if ax else plt.axes()

    # calculate averages up front to avoid repeated calculations
    y_avg = data['y'].mean()
    x_avg = data['x'].mean()

    # set x limits
    adj_x = max((data['x'].max() - x_avg), (x_avg - data['x'].min())) * 1.1
    lb_x, ub_x = (x_avg - adj_x, x_avg + adj_x)
    ax.set_xlim(lb_x, ub_x)

    # set y limits
    adj_y = max((data['y'].max() - y_avg), (y_avg - data['y'].min())) * 1.1
    lb_y, ub_y = (y_avg - adj_y, y_avg + adj_y)
    ax.set_ylim(lb_y, ub_y)

    # set x tick labels
    if xtick_labels:
        ax.set_xticks([(x_avg - adj_x / 2), (x_avg + adj_x / 2)])
        ax.set_xticklabels(xtick_labels)

    # set y tick labels
    if ytick_labels:
        ax.set_yticks([(y_avg - adj_y / 2), (y_avg + adj_y / 2)])
        ax.set_yticklabels(ytick_labels, rotation='vertical', va='center')

    # plot points and quadrant lines
    ax.scatter(x=data['x'], y=data['y'], c='lightblue', edgecolor='darkblue',
    zorder=99)
    ax.axvline(x_avg, c='k', lw=1)
    ax.axhline(y_avg, c='k', lw=1)



def plot_classes(featx_1,featx_2,featx_3,featx_4,
                featy_1,featy_2,featy_3,featy_4):
    for i, j in zip(featx_1,featy_1):
           plt.scatter((j.flatten()),(i.flatten()), c='k')

    for i, j in zip(featx_2,featy_2):
           plt.scatter((j.flatten()),(i.flatten()), c='b')

    for i, j in zip(featx_3,featy_3):
           plt.scatter((j.flatten()),(i.flatten()), c='r')
        
    for i, j in zip(featx_4,featy_4):
           plt.scatter((j.flatten()),(i.flatten()), c='g')



    exercise = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=10, label='Exercise')
    rest = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=10, label='Rest')
    mental = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=10, label='Mental')
    breath = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                          markersize=10, label='Breath')

    plt.legend(handles=[exercise, rest, mental,breath])
    plt.xlabel('HF')
    plt.ylabel('LF')
    plt.savefig('hf_lf_hilbert.jpeg')


def plot_data_section(section, plot_val, data):
    data_new=data[data.section==section]
    data_new = data_new[data_new.fs>=1]
    users = list(data_new.user.unique())
    users_sample =  sample_users(users)
    plt.subplot(2,2,1)
    data_1 =  data[data.user==users_sample[0]]
    plt.plot(data_1[data_1.section==section][plot_val].to_numpy()[0])
    plt.subplot(2,2,2)
    data_2 =  data[data.user==users_sample[1]]
    plt.plot(data_2[data_2.section==section][plot_val].to_numpy()[0])
    plt.subplot(2,2,3)
    data_3 =  data[data.user==users_sample[2]]
    plt.plot(data_3[data_3.section==section][plot_val].to_numpy()[0])
    plt.subplot(2,2,4)
    data_4 =  data[data.user==users_sample[3]]
    plt.plot(data_4[data_4.section==section][plot_val].to_numpy()[0])
    return

def plot_test(feat1, model1, labels, plot_fig, save_fig):
        if plot_fig:
            fig, (ax1) = plt.subplots(nrows=1, ncols=1,figsize=(4,4))
    
            X0, X1 = feat1[:, 0], feat1[:, 1]
            xx, yy = make_meshgrid(X0, X1)
            plot_contours(ax1, model1, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
            ax1.scatter(X0, X1, c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            ax1.set_ylabel('LF(ia)')
            ax1.set_xlabel('HF(ia)')
            
            if save_fig:
                plt.savefig('svm_4class.jpg')
        return
       
def plot_data_grid(section, data):
    users = list(data.user.unique())
    users_sample =  sample_users(users)
    plt.subplot(2,2,1)
    data_1 =  data[data.user==users_sample[0]]
    plt.plot(data_1[data_1.section==section].rr_array.to_numpy()[0])
    plt.ylim([400, 1600])
    plt.subplot(2,2,2)
    data_2 =  data[data.user==users_sample[1]]
    plt.plot(data_2[data_2.section==section].rr_array.to_numpy()[0])
    plt.ylim([400, 1600])
    plt.subplot(2,2,3)
    data_3 =  data[data.user==users_sample[2]]
    plt.plot(data_3[data_3.section==section].rr_array.to_numpy()[0])
    plt.ylim([400, 1600])
    plt.subplot(2,2,4)
    data_4 =  data[data.user==users_sample[3]]
    plt.plot(data_4[data_4.section==section].rr_array.to_numpy()[0])
    plt.ylim([400, 1600])
    return


# plasma does not exist in matplotlib < 1.5
cmap = getattr(cm, "plasma_r", cm.hot_r)


def create_axes(title, figsize=(16, 6)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # define the axis for the first plot
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # define the axis for the zoomed-in plot
    left = width + left + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    # define the axis for the colorbar
    left, width = width + left + 0.13, 0.01

    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)

    return (
        (ax_scatter, ax_histy, ax_histx),
        (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
        ax_colorbar,
    )


def plot_distribution(axes, X, y, hist_nbins=50, title="", x0_label="", x1_label=""):
    ax, hist_X1, hist_X0 = axes

    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)

    # The scatter plot
    colors = cmap(y)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker="o", s=5, lw=0, c=colors)

    # Removing the top and the right spine for aesthetics
    # make nice axis layout
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    # Histogram for axis X1 (feature 5)
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(
        X[:, 1], bins=hist_nbins, orientation="horizontal", color="grey", ec="grey"
    )
    hist_X1.axis("off")

    # Histogram for axis X0 (feature 0)
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(
        X[:, 0], bins=hist_nbins, orientation="vertical", color="grey", ec="grey"
    )
    hist_X0.axis("off")


def make_plot(item_idx):
    title, X = distributions[item_idx]
    ax_zoom_out, ax_zoom_in, ax_colorbar = create_axes(title)
    axarr = (ax_zoom_out, ax_zoom_in)
    plot_distribution(
        axarr[0],
        X,
        y,
        hist_nbins=200,
        x0_label=feature_mapping[features[0]],
        x1_label=feature_mapping[features[1]],
        title="Full data",
    )

    # zoom-in
    zoom_in_percentile_range = (0, 99)
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)

    non_outliers_mask = np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) & np.all(
        X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1
    )
    plot_distribution(
        axarr[1],
        X[non_outliers_mask],
        y[non_outliers_mask],
        hist_nbins=50,
        x0_label=feature_mapping[features[0]],
        x1_label=feature_mapping[features[1]],
        title="Zoom-in",
    )

    norm = mpl.colors.Normalize(y_full.min(), y_full.max())
    mpl.colorbar.ColorbarBase(
        ax_colorbar,
        cmap=cmap,
        norm=norm,
        orientation="vertical",
        label="Color mapping for values of y",
    )