import os
import sys
sys.path.append('../helper_functions')
import argoinput_helpers  # ---------- EXTERNAL FUNCTIONS argoinput_helpers.py
import argparse
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset as NetCDFFile
import numpy as np


# HYCOM-ECOSMO MODEL FIELD'S UNITS
primprod_model_unit= '($mg.m^{-3}$)'
secprod_model_unit= '($mg.m^{-3}$)'
phyplktn_model_unit= '($mg.m^{-3}$)'
zooplktn_model_unit= '($mg.m^{-3}$)'
detri_model_unit= '($mg.m^{-3}$)'
nitr_model_unit= '($mg.m^{-3}$)'
pho_model_unit= '($mg.m^{-3}$)'
oxy_model_unit= "($µmol.kg^{-1}$)"

# BGC-ARGO FIELD'S UNITS
pres_argo_unit='(dbar)'
temp_argo_unit='($^{o}C$)'
sal_argo_unit='(PSU)'
oxy_argo_unit='($µmol.kg^{-1}$)'
chl_argo_unit='($mg.m^{-3}$)'
nit_argo_unit='($µmol.kg^{-1}$)'


# FIELD LABELS
oxy_label = "Dissolved Oxygen "
temp_label = "Temperature "
salin_label = "Salinity "
chl_label = "Chlorophyll-$\mathit{a}$ "
nit_label = "Nitrate "
pres_label = "Pressure "

def main(argo_number,if_temp,if_sal,if_oxy,if_nit,if_chl,if_topo):
   directory = "./"
   filename = directory + "GL_PR_PF_" + str(argo_number) + ".nc"
   nc = NetCDFFile(filename)  

   if argo_number is not None:
      print('')
      print('')
      print('-----------------------------------------------------------------------------------------------')
      print('START PROCESSING ARGO FLOAT:',argo_number)#, '. ', ' Looking for data : ', filename)
      print('-----------------------------------------------------------------------------------------------')
      
   number = int(argo_number)
   os.system('mkdir -p ../outputs/plot_variables' )
   OUTdirectory = "../outputs/plot_variables/"

   if if_temp: 
      # By default, we are reading QC flags 1, 5 and 8.
      pres, since, time2d, dates, lon, lonint, lat, latint, timeint, temp, tempqc = argoinput_helpers.readARGO(number, 'TEMP',all=True)
      if len(temp) != 0:
         flag_sum = np.sum(np.isin(tempqc, [1, 5, 8])) # Check the sum of QC flags 1, 5, and 8            
         if flag_sum == 0:
             print('CAUTION: No data available for QC flags 1, 5, and 8')
             print('         So, attempting to access data without QC flags 1, 5, and 8')
             print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')               
         if flag_sum != 0:
             print('Data available for QC flags 1, 5, and 8')
             temp = argoinput_helpers.construct(temp, tempqc, [1, 5, 8])
         else:
             flags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
             available_flags = [flag for flag in flags if np.sum(np.isin(tempqc, flag)) != 0]
             if len(available_flags) > 0:
                 print('Data available for QC flags:', available_flags)
             else:
                 print('No data available for any QC flag from 0 to 9')
                    
      datea,deptha,tempa=argoinput_helpers.interpol(time2d,pres,since,temp)
      argoinput_helpers.pmeshargo(tempa,deptha,datea,number,str('tempa')) 
      ppp = plt.gcf()
      ppp.canvas.print_figure(OUTdirectory+'temp'+'_'+str(number)+'.png',dpi=200)
      plt.close(ppp)

   if if_sal:
      pres, since, time2d, dates, lon, lonint, lat, latint, timeint, sal, salqc = argoinput_helpers.readARGO(number, 'PSAL',all=True)
      if len(sal) != 0:
         flag_sum = np.sum(np.isin(salqc, [1, 5, 8])) # Check the sum of QC flags 1, 5, and 8           
         if flag_sum == 0:
             print('CAUTION: No data available for QC flags 1, 5, and 8')
             print('         So, attempting to access data without QC flags 1, 5, and 8')
             print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')                
         if flag_sum != 0:
             print('Data available for QC flags 1, 5, and 8')
             sal = argoinput_helpers.construct(sal, salqc, [1, 5, 8])
         else:
             flags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
             available_flags = [flag for flag in flags if np.sum(np.isin(salqc, flag)) != 0]
             if len(available_flags) > 0:
                 print('Data available for QC flags:', available_flags)
             else:
                 print('No data available for any QC flag from 0 to 9')
                    
      datea,deptha,sala=argoinput_helpers.interpol(time2d,pres,since,sal)
      argoinput_helpers.pmeshargo(sala,deptha,datea,number,str('sala'),dbot=2000) 
      ppp = plt.gcf()
      ppp.canvas.print_figure(OUTdirectory+'sal'+'_'+str(number)+'.png',dpi=200)
      plt.close(ppp)

   if if_chl:
      pres, since, time2d, dates, lon, lonint, lat, latint, timeint, chl, chlqc = argoinput_helpers.readARGO(number, 'CPHL',all=True)
      if len(chl) != 0:        
        flag_sum = np.sum(np.isin(chlqc, [1, 5, 8])) # Check the sum of QC flags 1, 5, and 8        
        if flag_sum == 0:
            print('CAUTION: No data available for QC flags 1, 5, and 8')
            print('         So, attempting to access data without QC flags 1, 5, and 8')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')       
        if flag_sum != 0:
            print('Data available for QC flags 1, 5, and 8')
            chl = argoinput_helpers.construct(chl, chlqc, [1, 5, 8])
        else:
            flags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            available_flags = [flag for flag in flags if np.sum(np.isin(chlqc, flag)) != 0]
            if len(available_flags) > 0:
                print('Data available for QC flags:', available_flags)
            else:
                print('No data available for any QC flag from 0 to 9')
    
        datea,deptha,chla=argoinput_helpers.interpol(time2d,pres,since,chl)
        argoinput_helpers.pmeshargo(chla,deptha,datea,number,str('chla'),dbot=2000) 
        ppp = plt.gcf()
        ppp.canvas.print_figure(OUTdirectory+'chl'+'_'+str(number)+'.png',dpi=200)
        plt.close(ppp)

   if if_nit:
      pres, since, time2d, dates, lon, lonint, lat, latint, timeint, nit, nitqc = argoinput_helpers.readARGO(number, 'NTAW',all=True)
      if len(nit) != 0:        
         flag_sum = np.sum(np.isin(nitqc, [1, 5, 8])) # Check the sum of flags 1, 5, and 8           
         if flag_sum == 0:
             print('CAUTION: No data available for QC flags 1, 5, and 8')
             print('         So, attempting to access data without QC flags 1, 5, and 8')
             print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')       
         if flag_sum != 0:
             print('Data available for QC flags 1, 5, and 8')
             nit = argoinput_helpers.construct(nit, nitqc, [1, 5, 8])
         else:
             flags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
             available_flags = [flag for flag in flags if np.sum(np.isin(nitqc, flag)) != 0]
             if len(available_flags) > 0:
                 print('Data available for QC flags:', available_flags)
             else:
                 print('No data available for any QC flag from 0 to 9')
                
         datea,deptha,nita=argoinput_helpers.interpol(time2d,pres,since,nit)
         argoinput_helpers.pmeshargo(nita,deptha,datea,number,str('nita'),dbot=2000) 
         ppp = plt.gcf()
         ppp.canvas.print_figure(OUTdirectory+'nit'+'_'+str(number)+'.png',dpi=200)
         plt.close(ppp)

   if if_oxy:
      pres, since, time2d, dates, lon, lonint, lat, latint, timeint, oxy, oxyqc = argoinput_helpers.readARGO(number, 'DOX2',all=True)
      if len(oxy) != 0:       
         flag_sum = np.sum(np.isin(oxyqc, [1, 5, 8])) # Check the sum of QC flags 1, 5, and 8           
         if flag_sum == 0:
             print('CAUTION: No data available for QC flags 1, 5, and 8')
             print('         So, attempting to access data without QC flags 1, 5, and 8')
             print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')                
         if flag_sum != 0:
             print('Data available for QC flags 1, 5, and 8')
             oxy = argoinput_helpers.construct(oxy, oxyqc, [1, 5, 8])
         else:
             flags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
             available_flags = [flag for flag in flags if np.sum(np.isin(oxyqc, flag)) != 0]
             if len(available_flags) > 0:
                 print('Data available for QC flags:', available_flags)
             else:
                 print('No data available for any QC flag from 0 to 9')
                    
         datea, deptha, oxya = argoinput_helpers.interpol(time2d, pres, since, oxy)
         argoinput_helpers.pmeshargo(oxya,deptha,datea,number,str('oxya'),dbot=2000)
         ppp = plt.gcf()
         ppp.canvas.print_figure(OUTdirectory + 'oxy' + '_' + str(number) + '.png', dpi=200)
         plt.close(ppp)
        
   
   # plot trajectory
   if if_topo is not None:
      print('-----------------------------------------------------------------------------------')
      print('Plotting Trajectory ', 'from file :::> ' , filename,)
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      argoinput_helpers.trajectory(number,print=True,url=if_topo)
   else:
      print('-----------------------------------------------------------------------------------')
      print('Plotting Trajectory ', 'from file :::> ' , filename,)
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      argoinput_helpers.trajectory(number, 'TEMP',print=True)

        
   print('')
   print('--------------------------------------------------------------------------')
   print( 'ARGO FLOAT:',number, ' ', 'earliest date for model run is : ', dates[0])
   print( 'ARGO FLOAT:',number, ' ', 'latest date for model run is   : ', dates[-1])
   print('---------------------------------------------------------------------------')
   print('You need to decide on date_start and date_end')
   print('The format of these dates should be: 2000-01-01')
   print('')
   print('--------------------------------------------------------------------------------------------------------')
   print( 'SUCCESSFULLY PROCESSED ARGO FLOAT:',number, '. ', ' Look for plot(s) in  : ', OUTdirectory )
   print('--------------------------------------------------------------------------------------------------------')
   print('')
   print('') 
   
  

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('number',  type=str)
    parser.add_argument('--temp', action="store_true",default=False)
    parser.add_argument('--sal', action="store_true",default=False)
    parser.add_argument('--oxy', action="store_true",default=False)
    parser.add_argument('--nit', action="store_true",default=False)
    parser.add_argument('--chl', action="store_true",default=False)
    parser.add_argument('--topo', type=str)
    args = parser.parse_args()

    main(args.number,args.temp,args.sal,args.oxy,args.nit,args.chl,args.topo)
    
    ############################################################################
