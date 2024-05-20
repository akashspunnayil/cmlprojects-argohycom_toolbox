import sys
sys.path.append('../helper_functions')
import plot_argo_traj_fields
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc,rcParams
from netCDF4 import Dataset as NetCDFFile
from netCDF4 import Dataset, num2date
import xarray as xr
import datetime
import warnings
import math
import csv
import re
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from dateutil.relativedelta import relativedelta
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from cartopy.crs import PlateCarree
from cartopy.feature import COASTLINE, LAND
import pickle
import glob
import os
import os.path
import io
import cmocean
warnings.filterwarnings('ignore')

import numpy.ma as ma
from numpy import mean
from numpy import std
from numpy import cov
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import ttest_ind
import scipy.stats as stats
import scipy.stats
import goodness_of_fit

def readARGO(number, variable,suppress_print=False,*args, **kwargs):
    # file and variable names are compatible with Copernicus ARGO NetCDF files
    directory = "/media/akashs/FA22E72622E6E69B/DATA1_disk/1_NANSEN/DATA_LIBRARY_LOCAL/BGC_ARGO/arabian_sea/"
    filename = directory + "GL_PR_PF_" + str(number) + ".nc"
    nc = NetCDFFile(filename)         

    pres    = nc.variables['PRES'][:,:]
    presqc  = nc.variables['PRES_QC'][:,:]
    lon     = nc.variables['LONGITUDE'][:]
    lat     = nc.variables['LATITUDE'][:]
    time    = nc.variables['TIME'][:]
    deph    = nc.variables['DEPH'][:]
    
  
    if 'JULD' in nc.variables:
        IFREMER = True
        if variable == 'CPHL' : variable = 'CPHL'
        if variable == 'NTAW' : variable = 'NTAW'
        if variable == 'DOX2' : variable = 'DOX2'       
    else:
        IFREMER = False

    var = []; varqc = []
    
    if variable in nc.variables:
        var   = nc.variables[variable][:,:]
        varqc = nc.variables[variable+'_QC'][:,:]
        
        if IFREMER:
            var = np.ma.masked_where(presqc != b'1', var)
            pres = np.ma.masked_where(presqc != b'1', pres)          
#         else:
#             var = np.ma.masked_where(presqc != 1, var)
#             pres = np.ma.masked_where(presqc != 1, pres)

    if variable not in nc.variables: 
        print(variable + ' is not listed in the', filename)
        print('PROCESSING ARGO FLOAT ', number, ' WILL BE STOPPED')
        print('----------------------------------------------------------------')
        print(' ' )
        exit()
    else:
        if not suppress_print:
            print('Evaluating Field :', '|',variable,'|', 'from file :::> ' , filename,)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    if IFREMER:
        time    = nc.variables['JULD'][:]
    else:
        time    = nc.variables['TIME'][:]
    time2d = np.zeros((pres.shape))
    for z in range(pres.shape[1]):
        time2d[:,z] = time
    if IFREMER:
        since = int(nc.variables['JULD'].units[11:15])
    else:
        since = int(nc.variables['TIME'].units[11:15])
        if not suppress_print:
            print(  'since of ARGO', since   )
    dates = list()
    for t in range(time.shape[0]):
        dates.append( datetime.datetime(since,1,1) + datetime.timedelta(days=time[t]) )
    datestart = dates[0]
    dateend   = dates[-1]

    diff = datestart-datetime.datetime(since,1,1,0,0,0)
    hour1 = int(diff.days * 24. + diff.seconds/60./60.)
    diff = dateend-datetime.datetime(since,1,1,0,0,0)
    hour2 = int(diff.days * 24. + diff.seconds/60./60.)

    timerange = np.concatenate((float(hour1),time*24.),axis=None)
    timerange = np.concatenate((timerange,float(hour2)),axis=None)
    latrange = np.concatenate((lat[0],lat),axis=None)
    latrange = np.concatenate((latrange,lat[-1]),axis=None)
    lonrange = np.concatenate((lon[0],lon),axis=None)
    lonrange = np.concatenate((lonrange,lon[-1]),axis=None)
    latint = np.interp(np.arange(hour1,hour2+1,1),timerange,latrange)
    lonint = np.interp(np.arange(hour1,hour2+1,1),timerange,lonrange)
    timeint = np.arange(hour1,hour2+1,1)
     
    if not suppress_print:
        print('ARGO' + ' ' + str(number) + ' ' + str(variable) + ' minimum: ', int(var.min()) ) #int(var.min()) )
        print('ARGO' + ' ' + str(number) + ' ' + str(variable) + ' maximum: ', int(var.max()) )
#     print('ARGO' + ' ' + str(number) + ' ' + str(variable) + ' ' + 'shape ', var.shape)
    
    if kwargs.get('all'):
        return pres, since, time2d, dates, lon, lonint, lat, latint, timeint, var, varqc
    elif kwargs.get('simple'):
        return pres, time2d, dates, var, varqc
    else:
        return var, varqc


def select_flags(var,qc,flag, *args, **kwargs):
    IFREMER=False
    try :
        type(qc.max())
    except  Exception:
        IFREMER=True
    if IFREMER:
        flag = str(flag).encode()
    new = np.zeros(var.shape) - 1E10
    new[qc==flag] = var[qc==flag]
    new = np.ma.masked_where(new < -1E5, new)
    return new

def construct(var,qc,flags, *args, **kwargs):
    IFREMER=False
    try :
        type(qc.max())
    except  Exception:
        IFREMER=True
    new = np.zeros(var.shape) - 1E10
    n = np.size(flags)
    for i in range(n):
        if IFREMER:
            flag = str(flags[i]).encode()
            new[qc==flag] = var[qc==flag]
        else:
            new[qc==flags[i]] = var[qc==flags[i]]
    new = np.ma.masked_where(new < -1E5, new)
    return new


def interpol(time2d,depth,since,var):   
    time = time2d[:,0]
    x = list({int(d):d for d in time}.values()) 
    
    dates = list()
    for t in range(len(x)):
        dates.append( datetime.datetime(since,1,1) + datetime.timedelta(days=x[t]) )
    dates = np.array(dates) 

    d1d = depth.flatten()  # Flatten the "depth" array into a 1D array
    v1d = var.flatten()    # Flatten the "var" array into a 1D array
    t1d = time2d.flatten() # Flatten the "time2d" array into a 1D array

    t1d = t1d[~d1d.mask]   # Remove missing values from "t1d" based on the "mask" attribute of "d1d"
    v1d = v1d[~d1d.mask]   # Remove missing values from "v1d" based on the "mask" attribute of "d1d"
    d1d = d1d[~d1d.mask]   # Remove missing values from "d1d" based on its own "mask" attribute

    t1d = t1d[~v1d.mask]   # Remove any remaining missing values from "t1d" based on the "mask" attribute of "v1d"
    d1d = d1d[~v1d.mask]   # Remove any remaining missing values from "d1d" based on the "mask" attribute of "v1d"
    v1d = v1d[~v1d.mask]   # Remove any remaining missing values from "v1d" based on its own "mask" attribute
    
#     print('d1d 1 to 10   ', d1d[1:10])
    print('depth (d1d) max   ', d1d.max())

    if d1d.size == 0:  # check if d1d is empty
        print('CAUTION: The interpolation is not done due to d1d==0 ')
        print(' and returning original inputs instead interpolated parameters ')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#         return dates,-depth,var  # return None or some other appropriate value indicating that interpolation was not performed
    
    y = np.arange(0.,float(int(d1d.max())),0.5)     
    xm,ym = np.meshgrid(x,y)

    intp = griddata( (d1d,t1d), v1d, (ym,xm), method='nearest')

    return dates,-y,intp


def pmeshargo(fld,depth,dates,number,fld_name,*args, **kwargs): 
    fld[np.isnan(fld)] = 0
    if np.ma.is_masked(fld):
        fld = fld.data
    
    plt.style.use('seaborn')
    import cmocean

    if kwargs.get('first'):
       plotmin = datetime.datetime(first,1,1)
    else: 
       plotmin = dates[0]

    if kwargs.get('last'):
       plotmax = datetime.datetime(last,1,1)
    else: 
       plotmax = dates[-1]

    nyears = (plotmax - plotmin).days / 365.
    if nyears ==1 :
        interval=1
    elif nyears == 2 or nyears == 3:
        interval = 3
    else :
        interval = 6

    cmax = kwargs.get('cmax',None)
    cmax = fld.max() if cmax is None else cmax
    cmin = kwargs.get('cmin',None)
    cmin = fld.min() if cmin is None else cmin
    dtop = kwargs.get('dtop',None)
    dtop = (depth).max() if dtop is None else -dtop
    dbot = kwargs.get('dbot',None)
    dbot = (depth).min() if dbot is None else -dbot 

    cmapv = kwargs.get('cmapv',None)
    if cmapv != None:
       if cmapv == 'oxygen':
          cmap = cmocean.cm.oxy
       else:
          cmap = plt.cm.get_cmap('Spectral_r')
    else:
       cmap = plt.cm.get_cmap('Spectral_r')
    
    
    fld_label = ' '
    cb_unit   = ' '
    if fld_name=='oxya':
       fld_label = str(plot_argo_traj_fields.oxy_label)
       cb_unit   = str(plot_argo_traj_fields.oxy_argo_unit)
    elif fld_name=='tempa':
       fld_label = str(plot_argo_traj_fields.temp_label)
       cb_unit   = str(plot_argo_traj_fields.temp_argo_unit)
    elif fld_name=='sala':
       fld_label = str(plot_argo_traj_fields.salin_label)
       cb_unit   = str(plot_argo_traj_fields.sal_argo_unit)
    elif fld_name=='chla':
       fld_label = str(plot_argo_traj_fields.chl_label)
       cb_unit   = str(plot_argo_traj_fields.chl_argo_unit)
    elif fld_name=='nita':
       fld_label = str(plot_argo_traj_fields.nit_label)
       cb_unit   = str(plot_argo_traj_fields.nit_argo_unit)
            
    fig = plt.figure(figsize=(10,3.5),facecolor='w')
    ax  = fig.add_subplot(1,1,1)
    ax.set_position([0.125,0.28,0.75,0.65]) # [left, bottom, width, height] 
    pmesh = plt.pcolormesh(dates,depth,fld,cmap=cmap,shading='auto')
    plt.clim(cmin,cmax)
    ax.set_xlim([plotmin,plotmax])
    ax.set_ylim([dbot,dtop])

    fmt_year = mdates.MonthLocator(interval=interval)
    ax.xaxis.set_major_locator(fmt_year)
    # Minor ticks every month.
    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_minor_locator(fmt_month)
    # Text in the x axis will be displayed in 'YYYY-mm' format.
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    ax.set_title(f'{fld_label}{number}')
    ax.set_xlabel("Time (days)",fontsize=13)
    ax.set_ylabel("Depth (m)",fontsize=13)
    #ax.grid(which='major', axis='both',color='w')
    cbaxes = fig.add_axes([0.9, 0.35, 0.03, 0.5]) # [left, bottom, width, height]
    cb = plt.colorbar(pmesh, cax = cbaxes)
    cb.set_label(fld_label+cb_unit, rotation=270, fontsize=9, labelpad=20)
#     plt.tight_layout()    
    
def trajectory(number, variable, *args, **kwargs):
    directory = "/media/akashs/FA22E72622E6E69B/DATA1_disk/1_NANSEN/DATA_LIBRARY_LOCAL/BGC_ARGO/arabian_sea/"
    os.system('mkdir -p ../outputs/plot_variables' )
    OUTdirectory = "../outputs/plot_variables/"
    
    pres, since, time2d, dates, lon, lonint, lat, latint, timeint,var, varqc = readARGO(number, variable,suppress_print=True,all=True)

    time = time2d[:,0]
    time_rounded = (time - (datetime.datetime(1970, 1, 1) - datetime.datetime(since, 1, 1)).days).astype('datetime64[D]')

    filename = directory + "GL_PR_PF_" + str(number) + ".nc"
    nc = xr.open_dataset(filename)
    
    rc('font', weight='bold')
    fig = plt.figure(figsize=[8.75, 5])
    ax = plt.subplot(1, 1, 1, projection=PlateCarree())
    cmap = plt.get_cmap('Spectral_r')
    ax.set_position([0.09,0.125,0.7,0.8])

    ax.add_feature(LAND)
    ax.add_feature(COASTLINE, edgecolor='grey')
    
    ax.plot(nc.LONGITUDE[0],nc.LATITUDE[0],'r<', markersize=8)
    ax.plot(nc.LONGITUDE[-1],nc.LATITUDE[-1],'r>', markersize=8)
    
#     print('ax.get_extent() ---> ',ax.get_extent())

    ax.set_xlim([42, 80]);  ax.set_ylim([4, 28]) # set_xlim and set_ylim is more memory efficient than set_extent  
    gl=ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)  
    gl.xlabels_top = False; gl.ylabels_right = False; gl.xlines = False;  gl.ylines = False # manipulate `gridliner` object
       
    p = ax.scatter(nc.LONGITUDE,nc.LATITUDE,s=60,c=time_rounded,cmap=cmap,marker='.')
    cbaxes = fig.add_axes([0.8, 0.1, 0.02, 0.8])
    cb = plt.colorbar(p, cax = cbaxes)
    ax.set_facecolor('white') # set ocean color to white
    
    ax.set_title(f"Trajectory of WMO {str(nc.attrs['platform_code'])}",fontweight="bold")
    cb.set_label('Time (days)', rotation=270, fontsize=14, labelpad=18,fontweight="bold")
    ax.text(38.5,12,'Latitude (degress)', color='k',transform=PlateCarree(), rotation=90, fontsize=14,fontweight="bold")
    ax.text(55,1.5,'Longitude (degress)', color='k',transform=PlateCarree(), rotation=360, fontsize=14,fontweight="bold")
    
    fmt_qu_year = mdates.MonthLocator(interval=3)
    cbaxes.yaxis.set_major_locator(fmt_qu_year)
    fmt_month = mdates.MonthLocator()
    cbaxes.yaxis.set_minor_locator(fmt_month)
    cbaxes.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) 
        
#     fig.canvas.print_figure(OUTdirectory+"trajectory"+'_'+str(number)+".png",dpi=210)
    buf = io.BytesIO()
    fig.canvas.print_png(buf)
    with open(OUTdirectory+"trajectory"+'_'+str(number)+".png", 'wb') as f:
        f.write(buf.getbuffer())
    plt.tight_layout() # Add this line to adjust the spacing between subplots

#     plt.close(fig)

# getvar() function for extracting the model fields and dimensions                  
def getvar(nc,varib, *args, **kwargs):
    time0 = nc.variables['time'][:] 
    time_rounded = np.array(time0, dtype='datetime64[D]') # Round/re-arrange time by days
    time = str(time_rounded)

    # start Year
    sincey0 = str(time.split(' ')[0].split('-')[0])[2:]     # [2:] to pick 2nd character of " ['2011 " of " ['2011-12-17'] "    
    sincey = int(sincey0)                                   # converting to integer to supply values to datetime.timedelta()
    # start month
    sincem0 = str(time.split(' ')[0].split('-')[1])
    sincem = int(sincem0)                                   # converting to integer to supply values to datetime.timedelta()
    # start day
    sinced0 = str(time.split(' ')[0].split('-')[2])[0:2]    # [0:2] to pick last couple of character from date string. 
    sinced = int(sinced0)


    ## START define new 'time' ##
    time0 = nc.variables['time'] 
    time_rounded = np.array(time0, dtype='datetime64[s]') # Round/re-arrange time by days
    time = str(time_rounded)
    
    ## END define new 'time' ##

    depth = nc.variables['depth'][:]
    if varib == 'nuh':
        depth = nc.variables['zi'][:,:,0,0]

    timeall = np.zeros(( time0.shape[0],depth.shape[0] ))
    yearall = np.zeros(( time0.shape[0],depth.shape[0] ))
    year = np.zeros(( time0.shape[0] ))

    dates = list()
    for t in range(time0.shape[0]) :
        yearr = ( datetime.datetime(sincey, sincem, sinced) + t*datetime.timedelta(days=1))
        year[t] = int( yearr.strftime('%Y') )
        dates.append(yearr)

    first = kwargs.get('first',None)
    first = year[0] if first is None else first

    last = kwargs.get('last',None)
    last = year[-1] if last is None else last


#     yindex = np.array(np.where((year >= first) & (year <= last))) 
    yindex = np.intersect1d(np.where(year >= first), np.where(year <= last)) 
#     print('yindex', yindex)
#     print('yindex shape', yindex.shape)

#     for d in range(depth.shape[0]) :
#        timeall[:,d] = time

    if varib == 'time':
       var = np.copy(timeall)
    elif varib == 'date':
       var = np.copy(dates)
    elif varib == 'depth':
       var = np.copy(depth)
    elif varib == 'depthi':
       var = nc.variables['zi'][:,:,0,0]    
    elif varib == 'mld':
       var = nc.variables['mld_surf'][:,0,0]
    elif varib == 'chl' :
       var1 = nc.variables['ECO_diachl'][:,:,0,0]
       var2 = nc.variables['ECO_flachl'][:,:,0,0]
       var = var1 + var2
    else :
         var = nc.variables[varib][:,:,0]

    if varib == 'date':
       var = var[yindex]
    elif len(var.shape) == 2:
       var = var[yindex,:]
#        var = var[0,:]
#     elif len(var.shape) == 1:
#        var = var[yindex]
    if varib == 'ECO_oxy' :
#        var = var
       var = var.where(var>0, np.nan) # set any values in the 'var' that are less than or equal to 0 to NaN
#        var = var.where(var > 0| var.isnull(), 0) # HERE, I replaced the -ve values with 0. 
                                                 # But why the intermediate depths having such negative values
       print('Model' + ' ' + 'ECO_oxy' + ' ' + 'for ARGO ' \
             + ' ' + '(defined in function) minimum : ', int(var.min()))
       print('Model' + ' ' + 'ECO_oxy' + ' ' + 'for ARGO ' \
             + ' ' + '(defined in function) maximum : ', int(var.max()))
#        print('Model' + ' ' + 'ARGO'+' '+'ECO_oxy '+ 'shape' , var.shape)
    if varib == 'ECO_no3' :
       var = var / 12.01 / 6.625
       var = var[np.where(var>10e20)]=np.nan
#        var = var.where(var > 0| var.isnull(), 0)
    if varib == 'ECO_sil' :
       var = var / 12.01 / 6.625
       var = var[np.where(var>10e20)]=np.nan
#        var = var.where(var > 0| var.isnull(), 0)
    if varib == 'ECO_pho' : 
       var = var / 12.01 / 106.0
       var = var[np.where(var>10e20)]=np.nan
#        var = var.where(var > 0| var.isnull(), 0)
    if varib == 'ECO_primprod' :
       var = var * 86400.
       var = var[np.where(var>10e20)]=np.nan
#        var = var.where(var > 0| var.isnull(), 0)
    if varib == 'ECO_secprod' :
       var = var * 86400.
       var = var[np.where(var>10e20)]=np.nan
#        var = var.where(var > 0| var.isnull(), 0)

    return var
    

#def contourdata(x,y,z,title,print_xticklabel,print_yticklabel,number,*args, **kwargs):
def contourdata(x,y,z,title,print_xticklabel,print_yticklabel,number,currVar,*args, **kwargs):
    label = ""
    unit  = ""
    
    '''
    typical use:
    
    for ARGO:
    contourdata(timeargo,pres,flag_temp,nlevels=11,dbot=1000,sinced=[since,1,1,0,0,0],type='inferno') 

    for MODEL:
    contourdata(0,0,0,nlevels=11,dbot=1000,type='inferno',model=nc,variable='ECO_no3',sinced=0,minyear=2014,maxyear=2020)
    # default colormap = inferno, no need to use type='inferno' in that case
    '''
    import cmocean
    #plt.style.use('seaborn')

    model = kwargs.get('model',None)
    if model is not None:
       variable = kwargs.get('variable')

       ## START define new 'time' ##
       time0 = model.variables['time'][:] 
       time_rounded = np.array(time0, dtype='datetime64[D]') # Round/re-arrange time by days
       time = str(time_rounded)
#        print('time0 shape is ---------->', time0.shape)
       ## END define new 'time' ##
    
        # start year
       sincey0 = str(time.split(' ')[0].split('-')[0])[2:]  # [2:] to pick 2nd character of " ['2011 " of " ['2011-12-17'] " date string
       sincey = int(sincey0)                                # converting to integer to supply values to datetime.timedelta()
        # start month
       sincem0 = str(time.split(' ')[0].split('-')[1])
       sincem = int(sincem0)                                # converting to integer to supply values to datetime.timedelta()
        # start day
       sinced0 = str(time.split(' ')[0].split('-')[2])[0:2] # [0:2] to pick last couple of character from date string. 
       sinced = int(sinced0)                                # converting to integer to supply values to datetime.timedelta()

        
       time_model0 =model['time'].values
       time_model = str(time_model0)
       
       sinceh  = str(time_model.split('T')[1].split(':')[0])
       sincemi = str(time_model.split('T')[1].split(':')[1])
       sinces  = str(time_model.split('T')[1].split(':')[2].split('.')[0])
       sincedd = datetime.datetime(int(sincey),int(sincem),int(sinced),int(sinceh),int(sincemi),int(sinces))

       z = getvar(model,variable)
       if variable == 'nuh':
          y = -(getvar(model,'depthi'))
       else:
          y = getvar(model,'depth')
       x = getvar(model,'time')
       dates_model = model['time'].values
       print('START DATE and END DATE of ORIGINAL MODEL variable ', dates_model[0], dates_model[-1] )
    
       scale_factor = kwargs.get('scale_factor',None)
       if scale_factor is not None:
          z = z * float(scale_factor)
        
       xi = dates_model[:]
       yi = y[:]
       xm = x
       ym = y
       intp = z.T 
       
    else:
       xnum = int(z.shape[0]/1.) # Get the number of rows in z, and divide by 1. The result is cast to an integer.
       ynum = int(z.shape[1]/1.) # Get the number of columns in z, and divide by 1. The result is cast to an integer

       x = x.flatten() # Flatten the x, y, and z arrays into 1D arrays.
       y = y.flatten()
       z = z.flatten()

#        if hasattr(y, 'mask'):
       x = x[~y.mask]  # remove elements from x where corresponding elements in y are masked
       z = z[~y.mask]  # remove elements from z where corresponding elements in y are masked
       y = y[~y.mask]  # remove elements from y where y values are masked
       
#        if hasattr(z, 'mask'):
       x = x[~z.mask]  # remove elements from x where corresponding elements in z are masked
       y = y[~z.mask]  # remove elements from y where corresponding elements in z are masked
       z = z[~z.mask]  # remove elements from z where z values are masked
        
       xi = np.linspace(x.min(),x.max(),num=xnum) # create an array of xnum equally spaced values b/w the min and max vals of x
       yi = np.linspace(y.min(),y.max(),num=ynum) # create an array of ynum equally spaced values b/w the min and max vals of y
       xm,ym = np.meshgrid(xi,yi) # create a 2D grid of points by combining the elements of xi and yi
       
       intp = griddata( (x,y), z, (xm,ym), method='nearest') # perform nearest-neighbor interpolation of z onto the new 2D grid (xm, ym) using the griddata function. The resulting array intp contains the interpolated values of z on the grid.
    
    dtop = kwargs.get('dtop',None)
    dtop = (yi).max() if dtop is None else -dtop
    dbot = kwargs.get('dbot',None)
    dbot = (yi).min() if dbot is None else -dbot
    botloc = np.abs(yi-(dbot)).argmin()
    toploc = np.abs(yi-(dtop)).argmin()

    cmax = kwargs.get('cmax',None)
    cmax = intp[toploc:botloc,:].max() if cmax is None else cmax
    cmin = kwargs.get('cmin',None)
    cmin = intp[toploc:botloc,:].min() if cmin is None else cmin

#     cmap = plt.get_cmap('inferno')
    cmaptype = kwargs.get('cmaptype',None)
#     if cmaptype is not None:
    if cmaptype == 'balance':
       cmap = cmocean.cm.balance
    if cmaptype == 'algae':
       cmap = cmocean.cm.algae
    elif cmaptype == 'saline':
       cmap = cmocean.cm.saline
    elif cmaptype == 'solar':
       cmap = cmocean.cm.solar
    elif cmaptype == 'matter':
       cmap = cmocean.cm.matter
    elif cmaptype == 'amp':
       cmap = cmocean.cm.amp
    elif cmaptype == 'thermal':
       cmap = cmocean.cm.thermal
    elif cmaptype == 'inferno':
       cmap = plt.get_cmap('inferno')
    elif cmaptype == 'magma':
       cmap = plt.get_cmap('magma')
    elif cmaptype == 'spectral':
       cmap = plt.cm.get_cmap('Spectral_r')
    else:
       cmap = cmocean.cm.thermal

    #cmap = cmocean.cm.thermal if cmap is None else cmap #plt.get_cmap('Spectral_r')
    nlevels = kwargs.get('nlevels',None)
    nlevels = 10 if nlevels is None else nlevels
    levels = MaxNLocator(nbins=nlevels).tick_values( cmin,cmax )
#    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    sinced = kwargs.get('sinced',None)
    if sinced is not None:
       dates = list()
       if model is None:
          sincedd = datetime.datetime(int(sinced[0]),int(sinced[1]),int(sinced[2]),int(sinced[3]),int(sinced[4]),int(sinced[5]))
       for t in range(xi.shape[0]):
           if model is None:
              year = int((sincedd + datetime.timedelta(days=xi[t])).strftime('%Y'))
              month = int((sincedd + datetime.timedelta(days=xi[t])).strftime('%m'))
              day   = int((sincedd + datetime.timedelta(days=xi[t])).strftime('%d'))
              hours = int((sincedd + datetime.timedelta(days=xi[t])).strftime('%H'))
              mins  = int((sincedd + datetime.timedelta(days=xi[t])).strftime('%M'))
           else:
              year = pd.to_datetime(dates_model[t]).year
              month = pd.to_datetime(dates_model[t]).month
              day = pd.to_datetime(dates_model[t]).day
              hours = pd.to_datetime(dates_model[t]).hour
              mins = pd.to_datetime(dates_model[t]).minute           
           dates.append(datetime.datetime(year,month,day,hours,mins))

    nyears = dates[-1].year - dates[0].year + 1
    if nyears ==1 :
       interval=1
    elif nyears == 2 or nyears == 3:
       interval = 3
    else :
       interval = 6
        
    minyear = kwargs.get('minyear',None)
    maxyear = kwargs.get('maxyear',None)
    if maxyear is None: 
       plotmin=datetime.datetime(dates[0].year,1,1)
       plotmax=datetime.datetime(dates[-1].year+1,1,1,0,0,0)
    else:
       if isinstance(minyear, datetime.date):
           plotmin = datetime.datetime.strptime(str(minyear), '%Y-%m-%d')
           plotmax = datetime.datetime.strptime(str(maxyear), '%Y-%m-%d')
       else:
           plotmin=datetime.datetime(int(minyear),1,1)
           plotmax=datetime.datetime(int(maxyear),1,1,0,0,0)  
   
    if currVar == 'oxy':
       label = plot_argo_traj_fields.oxy_label
       unit = plot_argo_traj_fields.oxy_model_unit
    elif currVar == 'chl':
       label = plot_argo_traj_fields.chl_label
       unit = plot_argo_traj_fields.chl_argo_unit
    elif currVar == 'temp':
       label = plot_argo_traj_fields.temp_label
       unit = plot_argo_traj_fields.temp_argo_unit
    elif currVar == 'sal':
       label = plot_argo_traj_fields.sal_label
       unit = plot_argo_traj_fields.sal_argo_unit
    elif currVar == 'nit':
       label = plot_argo_traj_fields.nit_label
       unit = plot_argo_traj_fields.nit_argo_unit
    else:
          # Handle the case when currVar is neither 'oxy' nor 'chl'
       label = "Variable"
       unit = "Unit"
        
       
    fig = plt.figure(figsize=(8.5,3.5),facecolor='w')
    ax  = fig.add_subplot(1,1,1)
#     ax = plt.gca()
    
    isoline_min_depth = None
    isoline_max_depth = None
    isoline_upper_bound_range = None
    isoline_lower_bound_range = None
    
    ARGOinterpolation = kwargs.get('ARGOinterpolation', True)
    if ARGOinterpolation:
       print('  ')
       print('Interpolated ARGO variable will be plotted (CONTOUR PLOT) ')
       print('----------------------------------------------------------------')
       print('NOTE: If interpolation not required: call contourdata() with ARGOinterpolation=False  ')
       print('START DATE and END DATE of interpolated ARGO variable ', dates[0], dates[-1] )
       cs = plt.contourf(dates,-yi,intp,levels=levels,cmap=cmap)#,extend='both')
#        plt.colorbar(cs)
       cbar = plt.colorbar(cs) 
       plt.clim(cmin,cmax)
#        plt.clim(vmin=0, vmax=300)
       # Add a contour line for the value of 20
       
       isoline = kwargs.get('isoline', True)
       
       if isoline: 
          cs = plt.contour(dates, -yi, intp, levels=[5, 10, 20], colors=['green','red','white'], linewidths=1)
          # Get the coordinates of the contour line segments
          contour_lines = cs.allsegs[0]
          plt.clabel(cs, fmt='%d', fontsize=12, inline=True)
 
          
       if print_xticklabel:
          plt.xlabel('Time (days)',fontsize=15)
          ax.set_xticklabels(ax.get_xticks(), rotation = 45)
          fmt_half_year = mdates.MonthLocator(interval=interval)
          ax.xaxis.set_major_locator(fmt_half_year)
          fmt_month = mdates.MonthLocator()
          ax.xaxis.set_minor_locator(fmt_month)
          ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
          plt.xlim([plotmin,plotmax])
          plt.title(title,fontsize=11)     
       else:
          plt.setp( ax.get_xticklabels(), visible=False)
          plt.title(title,fontsize=15)
       if print_yticklabel:
          plt.ylabel('Depth (m)',fontsize=15)
    #       ax.set_yticklabels(ax.get_yticks(), fontsize=15) # This will set ylim to original as per input data
          plt.ylim([dbot,dtop])
       else:
          plt.setp( ax.get_yticklabels(), visible=False)
    #       plt.ylim([-dbot,0])

#        cbar.set_label(str(plot_argo_traj_fields.oxy_label)+str(plot_argo_traj_fields.oxy_model_unit), rotation=270, fontsize=8, labelpad=10,fontweight="bold")
       cbar.set_label(label + unit, rotation=270, fontsize=8, labelpad=10, fontweight="bold")
       plt.clim(cmin,cmax)
       cs.cmap.set_under(cmap(cmin))
       cs.cmap.set_over(cmap(cmax))
       plt.ylim([dbot,0])
       plt.tight_layout()
    else:
       print('  ')
       print('Non-Interpolated ARGO variable will be plotted (SCATTER PLOT)' )
       print('----------------------------------------------------------------')
    
       decimal_date = np.array(x)
       epoch = datetime.datetime(1950, 1, 1)
       if isinstance(decimal_date.tolist()[0], list):
            # List is nested, so flatten it
           flattened_list = [item for sublist in decimal_date.tolist() for item in sublist]
       else:
            # List is already flat
           flattened_list = decimal_date.tolist()
       delta = [datetime.timedelta(days=int(date)) for date in flattened_list]
       date = [epoch + d for d in delta]
       print('START DATE and END DATE of ORIGINAL ARGO variable ', dates[0], dates[-1] )  
       print('shapes of SCATTER date,-y,z', len(date),y.shape,z.shape)
       cs = plt.scatter(date,-y,c=z,s=1,cmap=cmap) 
       cbar = plt.colorbar(cs)
       
       plt.xticks(rotation=45)
       fmt_half_year = mdates.MonthLocator(interval=interval)
       fmt_month = mdates.MonthLocator()
       ax.xaxis.set_minor_locator(fmt_month)
        
       if len(x) < 100:
#           ax.xaxis.set_major_locator(mdates.MonthLocator())
          ax.xaxis.set_major_locator(fmt_half_year)
          ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
       else:
#           ax.xaxis.set_major_locator(mdates.YearLocator())
          ax.xaxis.set_major_locator(fmt_half_year)
          ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))       

       cbar.set_label(label + unit, rotation=270, fontsize=8, labelpad=10, fontweight="bold")
       plt.title(title,fontsize=11) 
       plt.ylabel('Depth (m)',fontsize=15)
       plt.xlabel('Time (days)',fontsize=15)
       plt.clim(cmin,cmax)
       cbar.cmap.set_under(cmap(0))
       cbar.cmap.set_over(cmap(300))
       plt.ylim([dbot,0])
       cs.set_array(z)
       plt.tight_layout()
             

    return intp, dates, -yi, isoline_min_depth, isoline_max_depth, isoline_upper_bound_range, isoline_lower_bound_range,label,unit
      


# interpolationg argo depths using model depths. bcoz argo depths are high resolution compared to model depth levels. 
# so it make sense reducing resolution of argo depth levels rather than increasing low resolution depth levels of model.
# interpolating low resolution model depths to high resolution argo depth levels is not a good idea.
def argo2model(refdep,argo,argodep,argotime,model,modeldep,modeltime):
    index1 = modeldep >= -refdep
    modeldata = model[index1,:]

    sec_argo = np.zeros((len(argotime)))
    for t in range(len(argotime)): 
        sec_argo[t] = (argotime[t] - datetime.datetime(1990,1,1)).seconds + (argotime[t] - datetime.datetime(1990,1,1)).days * 86400.
    sec_model = np.zeros((len(modeltime)))
    for t in range(len(modeltime)): 
        sec_model[t] = (modeltime[t] - datetime.datetime(1990,1,1)).seconds + (modeltime[t] - datetime.datetime(1990,1,1)).days * 86400.

    argodata1 = np.zeros((modeldata.shape[0],sec_argo.shape[0]))

    for t in range(len(argotime)):
        argodata1[:,t] = np.interp( modeldep[index1], np.flipud(argodep), np.flipud(argo[:,t]) )
   
    argodata2 = np.zeros((modeldata.shape[0],sec_model.shape[0]))
    for d in range(argodata1.shape[0]):
        argodata2[d,:] = np.interp( sec_model, sec_argo, argodata1[d,:] )

    return argodata2,index1


# def cont(dates,yi,intp,levels,ax,plotmin,plotmax,dbot,title,print_xticklabel,print_yticklabel):
def cont(dates,yi,intp,title,print_xticklabel,print_yticklabel,currVar,*args, **kwargs):
    
    label = ""
    unit = ""
    
    fig = plt.figure(figsize=(8.5,3.5),facecolor='w')
    ax  = fig.add_subplot(1,1,1)
#     ax = plt.gca()
    
    dtop = kwargs.get('dtop',None)
    dtop = (yi).max() if dtop is None else -dtop
    dbot = kwargs.get('dbot',None)
    dbot = (yi).min() if dbot is None else -dbot

    botloc = np.abs(yi-(-dbot)).argmin()
    toploc = np.abs(yi-(-dtop)).argmin()

    cmax = kwargs.get('cmax',None)
    cmax = intp[toploc:botloc,:].max() if cmax is None else cmax
    cmin = kwargs.get('cmin',None)
    cmin = intp[toploc:botloc,:].min() if cmin is None else cmin
    
    nlevels = kwargs.get('nlevels',None)
    nlevels = 10 if nlevels is None else nlevels
    levels = MaxNLocator(nbins=nlevels).tick_values( cmin,cmax )  
    
    dates = np.array(dates, dtype='datetime64')
    epoch = datetime.datetime(1950, 1, 1)
    for t in range(dates.shape[0]):
        date_obj = datetime.datetime.fromtimestamp(np.datetime64(dates[t], 'ns').tolist() / 1e9)
        days = (date_obj - epoch).days
        year = int((epoch + datetime.timedelta(days=days)).strftime('%Y'))
        
    cmaptype = kwargs.get('cmaptype',None)
    if cmaptype == 'balance':
        cmap = cmocean.cm.balance
    if cmaptype == 'algae':
        cmap = cmocean.cm.algae
    elif cmaptype == 'haline':
        cmap = cmocean.cm.haline
    elif cmaptype == 'solar':
        cmap = cmocean.cm.solar
    elif cmaptype == 'matter':
        cmap = cmocean.cm.matter
    elif cmaptype == 'phase':
        cmap = cmocean.cm.phase
    elif cmaptype == 'thermal':
        cmap = cmocean.cm.thermal
    elif cmaptype == 'inferno':
        cmap = plt.get_cmap('inferno')
    elif cmaptype == 'magma':
        cmap = plt.get_cmap('magma')
    elif cmaptype == 'spectral':
        cmap = plt.cm.get_cmap('Spectral_r')
    else:
        cmap = cmocean.cm.thermal

    datesy = pd.to_datetime(dates)
    nyears = datesy[-1].year - datesy[0].year + 1
    
    if nyears ==1 :
       interval=1
    elif nyears == 2 or nyears == 3:
       interval = 3
    else :
       interval = 6
    
    minyear = kwargs.get('minyear',None)
    maxyear = kwargs.get('maxyear',None)
        
    if maxyear is None: 
        plotmin=datetime.datetime(datesy[0].year,1,1)
        plotmax=datetime.datetime(datesy[-1].year+1,1,1,0,0,0)

    else:
        if isinstance(minyear, datetime.date):
            plotmin = datetime.datetime.strptime(str(minyear), '%Y-%m-%d')
            plotmax = datetime.datetime.strptime(str(maxyear), '%Y-%m-%d')
        else:
            plotmin=datetime.datetime(int(minyear),1,1)
            plotmax=datetime.datetime(int(maxyear),1,1,0,0,0) 
              
        
    APPLYscatter = kwargs.get('APPLYscatter', False)
    if APPLYscatter: 
        print('  ')
        # replicate date and y to have the same shape as z
        dates = np.transpose(np.tile(dates, (yi.shape[0], 1)) )
        yi = np.transpose(np.tile(yi[:, np.newaxis], (1, dates.shape[0])) )
        print('shapes of MODEL scatter dates, yi and intp', len(dates), yi.shape, intp.shape)

        cs2 = plt.scatter(dates, yi,c=intp.T,s=0.1,cmap=cmap) # s=10 
        print('MODEL variable in ARGO T and D will be plotted (SCATTER PLOT)' )
        print('--------------------------------------------------------------------')
        cbar = plt.colorbar(cs2)
#         cbar = plt.colorbar(cs2, ax=ax)
        
    else: 
        print('  ')
        print('Interpolated/Default MODEL variable will be plotted (CONTOUR PLOT)' )
        print('------------------------------------------------------------------------')
        cs2 = plt.contourf(dates,yi,intp,levels=levels,cmap=cmap)#,extend='both')
        cs2.collections[0].set_alpha(1)
        cbar = plt.colorbar(cs2)
        plt.clim(cmin,cmax)
#     print('print(cs.levels)', cs.levels)

    isoline_min_depth = None
    isoline_max_depth = None
    isoline_upper_bound_range = None
    isoline_lower_bound_range = None
    
    isoline = kwargs.get('isoline', True)
    if isoline:
        cs2 = plt.contour(dates, yi, intp, levels=[5, 10, 20], colors=['green','red','white'], linewidths=1)
        cs2.collections[0].set_zorder(1)
        cs2.collections[0].set_alpha(1)
        
        # Get the coordinates of the contour line segments
        contour_lines = cs2.allsegs[0]
        
        # Extract the depths at level 20
        isoline = cs2.levels
        # Find the minimum and maximum depth values

        isoline_min_depth = contour_lines[0][:, 1].max() # In ocean the min depth is the max value (depths are -ve values)
        isoline_max_depth = contour_lines[-1][:, 1].min() # In ocean the max depth is the min value (depths are -ve values)
        #print('Depth values at level', isoline[0], ' : ', isoline_min_depth, 'meters and', isoline_max_depth, 'meters')
        
        isoline_upper_bound_range  = [contour_lines[0][:, 1].min(), contour_lines[0][:, 1].max()]
        isoline_lower_bound_range  = [contour_lines[-1][:, 1].min(),contour_lines[-1][:, 1].max()]
        
        plt.clabel(cs2, fmt='%.2f', fontsize=12, inline=True)   
    
#     ax.set_xticklabels(ax.get_xticks(), rotation = 45)
    plt.xticks(rotation=45)
    fmt_half_year = mdates.MonthLocator(interval=interval)
    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_minor_locator(fmt_month)

    if len(dates) < 100:
#           ax.xaxis.set_major_locator(mdates.MonthLocator())
       ax.xaxis.set_major_locator(fmt_half_year)
       ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
#           ax.xaxis.set_major_locator(mdates.YearLocator())
       ax.xaxis.set_major_locator(fmt_half_year)
       ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))       

    if currVar == 'oxy':
       label = plot_argo_traj_fields.oxy_label
       unit = plot_argo_traj_fields.oxy_model_unit
    elif currVar == 'chl':
       label = plot_argo_traj_fields.chl_label
       unit = plot_argo_traj_fields.chl_argo_unit
    elif currVar == 'temp':
       label = plot_argo_traj_fields.temp_label
       unit = plot_argo_traj_fields.temp_argo_unit
    elif currVar == 'sal':
       label = plot_argo_traj_fields.sal_label
       unit = plot_argo_traj_fields.sal_argo_unit
    elif currVar == 'nit':
       label = plot_argo_traj_fields.nit_label
       unit = plot_argo_traj_fields.nit_argo_unit
    else:
       # Handle the case when currVar is neither 'oxy' nor 'chl'
       label = "Variable"
       unit = "Unit"

    cbar.set_label(label + unit, rotation=270, fontsize=8, labelpad=10, fontweight="bold")    
    
#     cbar.set_label(str(plot_argo_traj_fields.oxy_label)+str(plot_argo_traj_fields.oxy_model_unit), rotation=270, fontsize=8, labelpad=10,fontweight="bold")
    plt.title(title,fontsize=11) 
    plt.ylabel('Depth (m)',fontsize=15)
    plt.xlabel('Time (days)',fontsize=15)
    plt.clim(cmin,cmax)
    cbar.cmap.set_under(cmap(cmin))
    cbar.cmap.set_over(cmap(cmax))
    plt.xlim([plotmin,plotmax])
    plt.ylim([dbot,0])
    plt.tight_layout()
      
    return cs2, isoline_min_depth, isoline_max_depth, isoline_upper_bound_range, isoline_lower_bound_range,label,unit


def slice_model(dbot,Mdate0, Mdep0, Adate0, Adep0, Model_oxy0, flag_oxy):
    Mdate0 = pd.to_datetime(Mdate0)
    Adate0 = pd.to_datetime(Adate0)
    Mdate0_formatted = Mdate0.strftime('%Y-%m-%d')
    Adate0_formatted = Adate0.strftime('%Y-%m-%d')
       
    # find the common dates between Mdate0 and Adate0
    common_indices = np.in1d(Mdate0_formatted, Adate0_formatted)
    Mdate0_sliced = Mdate0_formatted[common_indices]

    # slice Model_oxy0 to get subset that corresponds to the range of Adate0
    Model_oxy0_sliced = Model_oxy0[common_indices, :]

    # Define the depth levels for interpolation
    interp_depth = Adep0 
    # Initialize the Model_oxy0_interp array with NaN values
    Model_oxy0_interp = np.full((len(interp_depth), len(Mdate0)), np.nan)    
    for i in np.where(common_indices)[0]:
        f = interp1d(Mdep0, Model_oxy0[i, :], kind='linear', bounds_error=False, fill_value=np.nan)
        Model_oxy0_interp[:, i] = f(interp_depth)
    
    # Interpolate each time slice of Model_oxy0_sliced to interp_depth
    Model_oxy0_slice = np.zeros((len(interp_depth), Model_oxy0_sliced.shape[0]))
    for i in range(Model_oxy0_sliced.shape[0]):
        f = interp1d(Mdep0, Model_oxy0_sliced[i, :], kind='linear', bounds_error=False, fill_value=np.nan)
        Model_oxy0_slice[:, i] = f(interp_depth)

    # mask the corresponding values in Model_oxy0_slice with the masked values in flag_oxy
    Model_oxy0_slice_masked = np.ma.masked_where(flag_oxy.mask, Model_oxy0_slice.T)

    return Model_oxy0_interp, interp_depth, Model_oxy0_slice, Model_oxy0_slice_masked,Mdate0_sliced


# def scatter_n_stat(model,argo,fill_depths,print_xticklabel,print_yticklabel,xlabel,ylabel,titleSCA,item,*args, **kwargs):
def scatter_n_stat(model,argo,print_xticklabel,print_yticklabel,xlabel,ylabel,titleSCA,save_to_csv,outdir,item,all_items_str=None,*args, **kwargs):
#     not_nan_mask = ~np.isnan(model) & ~np.isnan(argo)
#     model = model[not_nan_mask]
#     argo  = argo[not_nan_mask]

    print('shapes of input model and argo', model.shape, argo.shape)

    # Assuming model and argo are your NumPy arrays
    if (np.isnan(model).any() or np.isinf(model).any()) or (np.isnan(argo).any() or np.isinf(argo).any()):
        not_nan_mask = (~np.isnan(model) & ~np.isnan(argo) & (model != np.inf) & (argo != np.inf))
        model_non_nan = model[not_nan_mask]
        argo_non_nan = argo[not_nan_mask]
    else:
        not_nan_mask = (~np.isnan(model) & ~np.isnan(argo) & (model != np.inf) & (argo != np.inf))
        model_non_nan = model
        argo_non_nan = argo
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # SUMMARIZE STATISTICS
    print('  ')
    print('----------------------------------------------------------')
    print('shapes of model and argo for statistics : ', model.shape, argo.shape)
    if all_items_str:
        print('SOME STATISTICS OF BGC-ARGO vs MODEL', '|', all_items_str , '|')
    else:
        print('SOME STATISTICS OF BGC-ARGO vs MODEL', '|', item , '|')
    print('----------------------------------------------------------')
    print(' ')
    print('\\','model: MEAN=%.3f STANDARD DEVIATION=%.3f' % (mean(model), std(model)))
    print('\\','argo : MEAN=%.3f STANDARD DEVIATION=%.3f' % (mean(argo), std(argo)))
    

    # calculate covariance matrix
    CoV = cov(model_non_nan, argo_non_nan)

    # calculate Pearson's correlation
    pearson_r, pearson_pvalue = pearsonr(model_non_nan, argo_non_nan) #the smallest possible 64bit floating point number is 2.225e-308
    t_statistic, p_value_TTEST = ttest_ind(model_non_nan, argo_non_nan)
    #     pearson_pvalue = "{:.10e}".format(pearson_pvalue)
    from decimal import Decimal
    pvalue_decimal = Decimal(str(pearson_pvalue))

    #     # calculate Spearman's correlation
    spearman_r, spearman_pvalue = spearmanr(model_non_nan, argo_non_nan)#the smallest possible 64bit floating point number is 2.225e-308
    #     print('\\','Spearman correlation between model_non_nan and argo_non_nan: %.3f' % spearman_r)
    #     print('\\','Probability value (p-value) between model_non_nan and argo_non_nan: %.3f' % spearman_pvalue)      

    # Calculate the slope and y-intercept of the linear regression line
    slope, intercept = np.polyfit(model_non_nan, argo_non_nan, 1)
    # Calculate the predicted y-values of the regression line
    argo_non_nan_pred = slope * model_non_nan + intercept
    # Calculate the total sum of squares
    ss_tot = np.sum((argo_non_nan - np.mean(argo_non_nan))**2)
    # Calculate the residual sum of squares
    ss_res = np.sum((argo_non_nan - argo_non_nan_pred)**2)
    # Calculate the coefficient of determination (r-squared)
    r_squared = 1 - (ss_res / ss_tot)  

    # Calculate the RMSE (y_pred - y_true)
    rmse = np.sqrt(np.mean((model_non_nan - argo_non_nan) ** 2))
    
    # Calculate the NRMSE (y_pred - y_true)
    nrmse = goodness_of_fit.nrmse(model_non_nan, argo_non_nan)
    
   # Calculate the Mean Error (ME)
    me = goodness_of_fit.me(model_non_nan, argo_non_nan)
    
    # Calculate the Index of Agreement (d)
    d = goodness_of_fit.d(model_non_nan, argo_non_nan)
   
    # Calculate the Modified Index of Agreement (md)
    md = goodness_of_fit.md(model_non_nan, argo_non_nan)
    
    # Calculate the Relative Index of Agreement (rd)
    rd = goodness_of_fit.rd(model_non_nan, argo_non_nan)
    
    # Calculate the Nash-sutcliffe Efficiency (nse)
    nse = goodness_of_fit.nse(model_non_nan, argo_non_nan)
    
    # Calculate the Modified Nash-sutcliffe Efficiency (mnse)
    mnse = goodness_of_fit.mnse(model_non_nan, argo_non_nan)
   
    # Calculate the Relative Nash-sutcliffe Efficiency (rnse)
    rnse = goodness_of_fit.rnse(model_non_nan, argo_non_nan)
   
    # Calculate the Kling Gupta Efficiency (kge)
    kge = goodness_of_fit.kge(model_non_nan, argo_non_nan)
     
    # Calculate the Deviation of gain (dg)
    dg = goodness_of_fit.dg(model_non_nan, argo_non_nan)
   
    # Calculate the Standard deviation of residual (sdr)
    sdr = goodness_of_fit.sdr(model_non_nan, argo_non_nan)
  
    # Calculate Mean Absolute Error (MAE) assume y_true and y_pred are numpy arrays of the same shape
    mae = np.mean(np.abs(argo_non_nan - model_non_nan))
   
    # Mean Bias Error (MBE)
    mbe = np.mean(model_non_nan - argo_non_nan)
   
    # Anomaly Correlation Coefficient (ACC)
    argo_non_nan_mean = np.mean(argo_non_nan)
    model_non_nan_mean = np.mean(model_non_nan)
    argo_non_nan_anomalies = argo_non_nan - argo_non_nan_mean
    model_non_nan_anomalies = model_non_nan - model_non_nan_mean
    acc, _ = pearsonr(argo_non_nan_anomalies, model_non_nan_anomalies)
    
    # Calculate the Model Efficiency (ME) # Samuelsen et al., 2015
    ME = 1 - (np.var(model_non_nan - argo_non_nan) / np.var(argo_non_nan))
#     ME = 1 - (np.sum((model - argo) ** 2) / np.sum((argo - np.mean(argo)) ** 2))


    # Bias 
    bias = np.sum(model_non_nan - argo_non_nan) / np.sum(argo_non_nan)
    
    # Calculate Percentage model_non_nan bias (Pbias) # Samuelsen et al., 2015
    Pbias = 100 * np.sum(model_non_nan - argo_non_nan) / np.sum(argo_non_nan)

    print(' ')
    print(' ')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print("{:<50} {:<50}".format(f"Metric {xlabel} Vs {ylabel}", "Value"))
    print("="*70)
    print("{:<50} {:.3f}".format("Pearson correlation (r):", pearson_r))
#     print("{:<50} {:.3e}".format("Probability value (p-value)", pvalue_decimal))
    print("{:<50} {:.3e}".format("Probability value (p-value):", pearson_pvalue))
    print("{:<50} {:.3f}".format("Spearman correlation (r):", spearman_r))
    print("{:<50} {:.3e}".format("Probability value (p-value):", spearman_pvalue))
    print("{:<50} {:.3f}".format("R-squared (r2):", r_squared))
    print("{:<50} {:.3f}".format("Root Mean Square Error (RMSE):", rmse))
    print("{:<50} {:.3f}".format("Normalised Root Mean Square Error (NRMSE):", nrmse))
    print("{:<50} {:.3f}".format("Mean Error (ME):", me))
    print("{:<50} {:.3f}".format("Index of Agreement (d):", d))
    print("{:<50} {:.3f}".format("Modified Index of Agreement (md):", md))
    print("{:<50} {:.3f}".format("Relative Index of Agreement (rd):", rd))
    print("{:<50} {:.3f}".format("Kling Gupta Efficiency (KGE):", kge))
    print("{:<50} {:.3f}".format("Deviation of gain (DG):", dg))
    print("{:<50} {:.3f}".format("Standard deviation of residual (SDR):", sdr))
    print("{:<50} {:.3f}".format("Mean Absolute Error (MAE):", mae))
    print("{:<50} {:.3f}".format("Mean Bias Error (MBE):", mbe))
    print("{:<50} {:.3f}".format("Anomaly Correlation Coefficient (ACC):", acc))
    print("{:<50} {:.3f}%".format("Nash-Sutcliffe Efficiency (NSE):", nse))
    print("{:<50} {:.3f}%".format("Modified Nash-Sutcliffe Efficiency (MNSE):", mnse))
    print("{:<50} {:.3f}%".format("Relative Nash-Sutcliffe Efficiency (RNSE):", rnse))
    print("{:<50} {:.3f}%".format("Model Efficiency (ME):", ME))
    print("{:<50} {:.3f}%".format("Percentage model bias (Pbias):", Pbias))
    print("{:<50} {:.3f}".format("Bias (bias):", bias))
    print(' ','--->  NOTE: IF Pbias is -ve, model is underestimating the field,\n ELSE overestimating [eqn(model - obs)]' )
    print(' ')
    print(' ')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%') 
        
#############################################################################################################################   

    # Organize metrics and values into a list of tuples
    metrics_values = [
        ("Pearson correlation (r):", pearson_r),
        ("Probability value (p-value):", pearson_pvalue),
        ("Spearman correlation (r):", spearman_r),
        ("Probability value (p-value):", spearman_pvalue),
        ("R-squared (r2):", r_squared),
        ("Root Mean Square Error (RMSE):", rmse),
        ("Normalised Root Mean Square Error (NRMSE):", nrmse),
        ("Mean Error (ME):", me),
        ("Index of Agreement (d):", d),
        ("Modified Index of Agreement (md):", md),
        ("Relative Index of Agreement (rd):", rd),
        ("Kling Gupta Efficiency (KGE):", kge),
        ("Deviation of gain (DG):", dg),
        ("Standard deviation of residual (SDR):", sdr),
        ("Mean Absolute Error (MAE):", mae),
        ("Mean Bias Error (MBE):", mbe),
        ("Anomaly Correlation Coefficient (ACC):", acc),
        ("Nash-Sutcliffe Efficiency (NSE):", nse),
        ("Modified Nash-Sutcliffe Efficiency (MNSE):", mnse),
        ("Relative Nash-Sutcliffe Efficiency (RNSE):", rnse),
        ("Model Efficiency (ME):", ME),
        ("Percentage model bias (Pbias):", Pbias),
        ("Bias (bias):", bias)
    ]

    if save_to_csv:
        # Define the output file path
        output_file = outdir+f"metrics_{xlabel}_Vs_{ylabel}.csv"

        # Write the metrics and values to a CSV file
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow([f"Metric_{xlabel}_Vs_{ylabel}", "Value"])
            # Write each metric and value as a row
            for metric, value in metrics_values:
                writer.writerow([metric, value])

        print(f"Metrics and values saved to '{output_file}'")

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    
    fig = plt.figure(figsize=(5.5,5.5),facecolor='w')
    ax  = fig.add_subplot(1,1,1)

    cs2 = ax.scatter(model, argo)
#     cs2 = px.scatter(model,argo, trendline="ols")
    
    if print_xticklabel:
        plt.xlabel(xlabel,fontsize=15)
        ax.set_xticklabels(ax.get_xticks(), rotation = 45, fontsize=15)#,fontsize=10);

        if all_items_str:
            plt.title(titleSCA.format(pearson_r=pearson_r, pearson_pvalue=pearson_pvalue, \
                                    r_squared=r_squared, rmse=rmse, item=item, all_items_str=all_items_str),fontsize=15)
        else:
            plt.title(titleSCA.format(pearson_r=pearson_r, pearson_pvalue=pearson_pvalue, \
                                    r_squared=r_squared, rmse=rmse, item=item),fontsize=15)
        
    else:
        plt.xlabel(xlabel,fontsize=15)
        ax.set_xticklabels(ax.get_xticks(), rotation = 0, fontsize=15)#,fontsize=10);
        plt.title(titleSCA.format(pearson_r=pearson_r, pearson_pvalue=pearson_pvalue, \
                                  r_squared=r_squared,rmse=rmse, item=item,all_items_str=all_items_str),fontsize=15)
    if print_yticklabel:
        plt.ylabel(ylabel,fontsize=15)
        ax.set_yticklabels(ax.get_yticks(), fontsize=15) # This will set ylim to original as per input data
    else:
        plt.setp( ax.get_yticklabels(), visible=False )
    
#     plt.gca().set_aspect('equal') 
#     plt.colorbar(cs2)
    plt.tight_layout()

    x, y = cs2.get_offsets().T
    
    return cs2,x,y,CoV,pearson_r,pearson_pvalue,spearman_r,spearman_pvalue 
    
    

class Tee:
    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, data):
        for output in self.outputs:
            output.write(data)

    def flush(self):
        for output in self.outputs:
            output.flush()
