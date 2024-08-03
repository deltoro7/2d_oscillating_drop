#!/usr/bin/env python3

import sys
import datetime
import os
import time as timer
import numpy as np
import pandas as pd
from pathlib import Path

# module matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['text.usetex'] = False #('text', usetex=False)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']

# module plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import itertools
import confidence_intervals as ci  # isort: skip

# environnement
sys.path.insert(0,os.environ['YALES2_HOME']+"/tools")

# --------------------
def new_job_indexes(total_time):
   return np.where(np.diff(total_time)<0)[0]+1

def format_number(variable, seuil_min=1.0E-1, seuil_max=9.0E3):
   ''' define tickformat and hoverformat for each plot
   '''

   # default value
   tickformat = '.3f'
   hoverformat = '.4f'

   # set specific format
   if abs(variable.max())<seuil_min or abs(variable.max())>seuil_max:
      tickformat = '.2e'
      hoverformat = '.4e'

   return tickformat, hoverformat

def parse_temporals(temporal_file,check_probe_names=True):
   data={}
   probenames=[]

   print(f"> Loading : {temporal_file}")

   # catch label (could be improved later)
   with open(temporal_file, 'r') as f:
      n_pts = 0
      for iline, line in enumerate(f):
         n_pts += 1
         if iline == 0:
            labels = list(filter(lambda x: not x=='',line.replace('#','').replace('\n','').split('    ')))
  
   n_data = len(labels)
   n_pts -= 1 # len(mylines)-1
   # print(n_data, n_pts)
   npdat = np.zeros((n_data,n_pts))
   with open(temporal_file, 'r') as f:
      next(f)
      for iline, line in enumerate(f):
         npdat[:,iline] = list(map(float,list(filter(lambda x: not x=='',line.replace('\n','').split(' ')))))
  
   for label in labels:
      fullname = label[label.index(':')+1::]
      position = int(label[0:label.index(':')])-1
      if '-' in fullname and check_probe_names:
         probename = fullname[0:fullname.index('-')]
         if probename not in probenames:
            probenames.append(probename)
            data[probename]={}
         dataname = fullname[fullname.index('-')+1::]
         data[probename][dataname]=npdat[position,:]
      else:
         data[fullname]=npdat[position,:]
   return data

def rawdata_to_df(data, drop_null_data: bool=True):
   ''' save rawdata to DataFrame
       return df
   '''

   # --------------------
   # transform data to pandas dataframe
   df = pd.concat([pd.DataFrame.from_dict(d) for d in data], axis=0)

   # --------------------
   # check if data is consistent to plot
   if (drop_null_data):
      for name in df.columns:
         # if sdt dev and mean value are close to 0, data is dropped
         if df[name].std()==0 and df[name].mean()==0:
            print(f'  {name} skipped. No data to plot.')
            df.drop(name, axis='columns', inplace=True)

         if "dt_gravity" in name or "dt_scalar_visc" in name:
            if df[name].mean()>1E10:
               df.drop(name, axis='columns', inplace=True)

   return df

def plot_temporal_data_in_pdf(df, temporal_filename, save_pdf, opt, print_iter_info=50):
   ''' plot temporal data to pdf file (with matplotlib)
   '''

   # --------------------
   # initialize pdf file settings
   pdf_filename = None
   if save_pdf:
      path_file = Path(temporal_filename).parent
      pdf_filename = Path(path_file)/(Path(temporal_filename).stem +"_plot.pdf")
      pdf = PdfPages(pdf_filename)

   # --------------------
   # split data by run sequences and save each sequence in list
   indexes = new_job_indexes(df['time'])
   dfs_splitted = np.split(df, indexes)
   
   # --------------------
   # loop over data to plot it (with filter every 'print_iter_info')
   jname = 1
   for iname, name in enumerate(df.columns):
   # for iname,name in enumerate(data.keys()):
      if opt=="all-in-page":
         fig = plt.figure(1,figsize=(16,12))
         plt.subplot(3,3,jname)
   
      # plot data
      for ix, df_splitted in enumerate(dfs_splitted):
         # filter data with print_iter_info
         df_tmp = df_splitted[(df_splitted.Niter%print_iter_info == 0) | (df_splitted.Niter==min(df_splitted.Niter)) | (df_splitted.Niter==max(df_splitted.Niter))]
         df_tmp.reset_index(drop=True, inplace=True)

         # plot
         xi = df_tmp['total_time']
         yi = df_tmp[name]

         jobnumber = f'{ix+1}'.zfill(3)
         label = f"RUN{jobnumber}"
         plt.plot(xi*1000.0,yi,label=label)

         tickformat, hoverformat = format_number(yi)
         plt.annotate(f"{yi.iloc[-1]:{hoverformat}}",xy=(1,yi.iloc[-1]),xytext=(8,0),xycoords=('axes fraction','data'),
                      textcoords='offset points',size=5,color=plt.gca().lines[-1].get_color())
      
      #plt.plot(data['total_time']*1000.0,data[name])
      ax = plt.gca()
      ax.set_xlabel('total_time [ms]')
      ax.set_ylabel(name)
      leg = plt.legend()
      leg.get_frame().set_alpha(0.5)
      fig.tight_layout()
      
      if opt=="all-in-page" and (jname%9==0 or iname==len(df.columns)-1):
         if save_pdf: pdf.savefig(bbox_inches="tight")
         plt.clf()
         jname = 1
      elif opt=="all-in-page":
         jname = jname + 1
      elif opt=="one-by-page":
         if save_pdf: pdf.savefig(bbox_inches="tight")
         plt.clf()
      else:
         print("Unknown PDF formating:",opt)
         sys.exit()
   
   # --------------------
   # Save the PDF file and show the figures
   if save_pdf:
      d = pdf.infodict()
      d['Title'] = 'post-processing'
      d['Author'] = os.environ['USER']
      d['Subject'] = 'Multipage pdf file containing automated post-processing'
      d['ModDate'] = datetime.datetime.today()
      pdf.close()
      print(f"  Pdf file saved in: {pdf_filename}")

   return pdf_filename.as_posix()

def plot_temporal_data_in_html(df, temporal_filename, save_html, print_iter_info=50):
   ''' plot temporal data to html file (with plotly)
   '''

   # --------------------
   # split data by run sequences and save each sequence in list
   indexes = new_job_indexes(df['time'])
   dfs_splitted = np.split(df, indexes)

   # --------------------
   # make figure
   bar_plots = ['Niter', 'time', 'total_Niter', 'total_time']
   column_names = [col for col in df.columns if not col in bar_plots]
   ncols = 3
   nrows = int(np.ceil((len(column_names) / ncols))) + 1
   fig_height = int(350*nrows) # 350 is the height of one row
   # print(nrows)

   fig = go.Figure()
   fig = make_subplots(rows=nrows, cols=ncols, horizontal_spacing=0.1)

   # --------------------
   # plot color settings
   col_pal = px.colors.qualitative.Plotly + px.colors.qualitative.Alphabet + px.colors.qualitative.Light24 + px.colors.qualitative.Dark24
   
   # initialize plot color iterator
   col_pal_iterator = itertools.cycle(col_pal)   

   # --------------------
   # configure bar plots
   jobnames = []
   for ix, df_splitted in enumerate(dfs_splitted):
      jobnumber = f'{ix+1}'.zfill(3)
      jobname = f"RUN{jobnumber}"
      jobnames.append(jobname)
      start = df_splitted['total_time'].to_list()[0]*1000.0
      niter = df_splitted['Niter'].to_list()[-1]
      total_niter = df_splitted['total_Niter'].to_list()[-1]
      time = df_splitted['time'].to_list()[-1]*1000.0
      total_time = df_splitted['total_time'].to_list()[-1]*1000.0

      # set trace color corresponding to job number
      trace_color = next(col_pal_iterator)
      
      # add trace
      fig.add_trace(go.Bar(x=[total_time-start], y=[jobname], base=start, orientation='h', marker_color=trace_color,
                           name=jobname, legendgroup=jobname, showlegend=True,
                           hovertemplate=
                              f"<b>{jobname}</b><br><br>" +
                              f"Niter: {niter:.0f}<br>" +
                              f"Total Niter: {total_niter:.0f}<br>" +
                              f"Time: {time:.2f} ms<br>" +
                              f"Begin total time: {start:.2f} ms<br>" +
                              f"End total time: {total_time:.2f} ms" +
                              "<extra></extra>",
                     ), row=1, col=1)

   # --------------------
   # configure others plots
   print(f'  Data are filtered every {print_iter_info} iterations')
   icol = 1 #0
   irow = 1
   for iname, name in enumerate(column_names):
      # --------------------
      # update increment
      iname +=1
      if iname%ncols==0 and iname!=0:
         irow += 1
         icol = 1
      else:
         icol += 1

      # --------------------
      # initialize color palette iterator
      col_pal_iterator = itertools.cycle(col_pal)

      # --------------------
      # loop over data to plot it (with filter every 'print_iter_info')
      for ix, df_splitted in enumerate(dfs_splitted):
         jobname = jobnames[ix]

         # set trace color corresponding to job number
         trace_color = next(col_pal_iterator)

         # filter data with print_iter_info
         df_tmp = df_splitted[(df_splitted.Niter%print_iter_info == 0) | (df_splitted.Niter==min(df_splitted.Niter)) | (df_splitted.Niter==max(df_splitted.Niter))]
         df_tmp.reset_index(drop=True, inplace=True)

         # add trace
         if not df_tmp.empty:
            fig.add_trace(go.Scatter(x=df_tmp['total_time']*1000.0, y=df_tmp[name], name=jobname,
                                     mode = "lines", line = dict(color = trace_color), legendgroup=jobname,
                                     showlegend=False,
                                    ), row=irow, col=icol)
               
            # update y axis
            tickformat, hoverformat = format_number(df_tmp[name])
            fig.update_yaxes(row=irow, col=icol, title=name,
                             tickformat=tickformat,
                             hoverformat=hoverformat,
                             linecolor = "#909497")

   # --------------------
   # update layout
   fig.update_layout(
      title="<br>" +
            '<span style="font-size: 16px;"> <b>Temporals plot</b> </span>' + '<br>' +
            f'<span style="font-size: 12px;">{Path(temporal_filename).resolve()}</span>',
      legend_traceorder='grouped',
      legend_tracegroupgap = 5,
      legend_title_text='RUNS : ',
      showlegend = True,
      font = dict(color = "#909497",size=10),
      plot_bgcolor = "white", # set the background colour
      height = fig_height,
      hovermode="x", # ['x', 'y', 'closest', False, 'x unified', 'y unified']
      hoverdistance=1, #100, # Distance to show hover label of data point
      spikedistance=1000, # Distance to show spike
      margin=dict(r=20, l=20, b=50, t=100),
   )

   # update x axis
   fig.update_xaxes(title = "total time [ms]", 
                    linecolor = "#909497",
                    showspikes=True, # Show spike line for X-axis
                    # Format spike
                    spikethickness=2,
                    spikedash="dot",
                    spikecolor="#999999",
                    spikemode="across",)

   # --------------------
   # save file if needed
   html_filename = None
   if save_html:
      path_file = Path(temporal_filename).parent
      html_filename = Path(path_file)/(Path(temporal_filename).stem +"_plot.html")
      fig.write_html(html_filename.as_posix(), include_plotlyjs="cdn", full_html=False)
      print(f"  Html file saved in: {html_filename}")

   return html_filename.as_posix()

def plot_estimators_in_html(df, temporal_filename, save_html, subsampling, stats_starting_time):
   ''' plot temporal data to html file (with plotly)
   '''

   # --------------------
   # split data by run sequences and save each sequence in list
   indexes = new_job_indexes(df['time'])
   dfs_splitted = np.split(df, indexes)

   # --------------------
   # make figure
   bar_plots = ['Niter', 'time', 'total_Niter', 'total_time']
   column_names = [col for col in df.columns if not col in bar_plots]
   ncols = 3
   nrows = 2*int(np.ceil((len(column_names) / ncols))+1)
   fig_height = int(350*nrows) # 350 is the height of one row
   # print(nrows)

   fig = go.Figure()
   fig = make_subplots(rows=nrows, cols=ncols, horizontal_spacing=0.1)

   # --------------------
   # plot color settings
   col_pal = px.colors.qualitative.Plotly + px.colors.qualitative.Alphabet + px.colors.qualitative.Light24 + px.colors.qualitative.Dark24
   
   # initialize plot color iterator
   col_pal_iterator = itertools.cycle(col_pal)   

   # --------------------
   # configure bar plots
   jobnames = []
   for ix, df_splitted in enumerate(dfs_splitted):
      jobnumber = f'{ix+1}'.zfill(3)
      jobname = f"RUN{jobnumber}"
      jobnames.append(jobname)

   # --------------------
   # configure others plots
   print(f'  Data are subsampled to process only {subsampling} iterations for confidence intervals estimation')
   print(f'  Statistics are computed from t={stats_starting_time} s')
   icol = 0 #0
   irow = 0
   for iname, name in enumerate(column_names):
      
      # --------------------
      # update increment
      iname +=1
      if iname%ncols==1:
         irow += 2
         icol = 1
      else:
         icol += 1

      # --------------------
      # initialize color palette iterator
      col_pal_iterator = itertools.cycle(col_pal)

      # --------------------
      # loop over data to plot it (with filter every 'print_iter_info')
      for ix, df_splitted in enumerate(dfs_splitted):
         jobname = jobnames[ix]

         # set trace color corresponding to job number
         trace_color = next(col_pal_iterator)

         # subsample data with subsampling
         print_iter_info = 1
         df_tmp = df_splitted[(df_splitted.Niter%print_iter_info == 0) | (df_splitted.Niter==min(df_splitted.Niter)) | (df_splitted.Niter==max(df_splitted.Niter))]
         df_tmp.reset_index(drop=True, inplace=True)

         time_plot = np.array(df_tmp['total_time'])
         subsampling = int(subsampling)
         time_interp = ci.get_time_interp(time_plot,subsampling)
         y = np.array(df_tmp[name])
         y_interp = np.interp(time_interp,time_plot,y)

         #reset data from stats_starting_time
         stats_starting_time = float(stats_starting_time)
         [time_interp,y_interp] = ci.reset_stats(time_interp,y_interp,stats_starting_time)

         [lags,autocorr,fit_auto_corr,a,b,sample_mean,sample_mean_err] = ci.compute_variance_sample_mean(y_interp,name)


         # add autocorrelation
         if not df_tmp.empty:
            fig.add_trace(go.Scatter(x=lags, y=autocorr, name=f'Autocorrelation of {jobname}',
                                     mode = "lines", line = dict(color = trace_color), legendgroup=jobname,
                                     showlegend=False,
                                    ), row=irow-1, col=icol)
            fig.add_trace(go.Scatter(x=lags, y=fit_auto_corr, name=f'exp(-(t/{a:.1f})**{b:.1f}) fit for {jobname}',
                                     mode = "lines", line = dict(color = "rgba( 255, 0, 0, 1.0)",dash="dot"), legendgroup=jobname,
                                     showlegend=False,
                                    ), row=irow-1, col=icol)
               
            # update y axis
            tickformat, hoverformat = format_number(df_tmp[name])
            fig.update_yaxes(row=irow-1, col=icol, title=f'Autocorrelation of {name}',
                             tickformat=tickformat,
                             hoverformat=hoverformat,
                             linecolor = "#909497")
            # update x axis
            fig.update_xaxes(row=irow-1, col=icol, title = "Iteration [-]", 
                    linecolor = "#909497",
                    showspikes=True, # Show spike line for X-axis
                    # Format spike
                    spikethickness=2,
                    spikedash="dot",
                    spikecolor="#999999",
                    spikemode="across",)
            
       # Compute variance of sample mean
         if not df_tmp.empty:
            # plot data
            fig.add_trace(go.Scatter(x=time_plot, y=y, name=name,
                                     mode = "lines", line={'color':"rgba( 0, 0, 0,0.3)"}, legendgroup=jobname,
                                     showlegend=False,
                                    ), row=irow, col=icol)
            fig.add_trace(go.Scatter(x=time_interp, y=sample_mean+sample_mean_err, name='upperbound',
                                     mode = "lines", legendgroup=jobname,
                                     showlegend=False,line={'color':"rgba( 0, 0, 0,0.1)"},
                                    ), row=irow, col=icol)
            fig.add_trace(go.Scatter(x=time_interp, y=sample_mean-sample_mean_err, name='lowerbound',
                                     mode = "lines", legendgroup=jobname,
                                     showlegend=False,fill='tonexty',fillcolor='lightpink',line={'color':"rgba( 0, 0, 0,0.1)"},
                                    ), row=irow, col=icol)
            fig.add_trace(go.Scatter(x=time_interp, y=sample_mean, name=f'sample mean of {name}',
                                     mode = "lines", line = dict(color = trace_color), legendgroup=jobname,
                                     showlegend=False,
                                    ), row=irow, col=icol)
            
               
            # update y axis
            tickformat, hoverformat = format_number(df_tmp[name])
            fig.update_yaxes(row=irow, col=icol, title=f'Sample mean of {name}',
                             tickformat=tickformat,
                             hoverformat=hoverformat,
                             linecolor = "#909497")
               # update x axis
            fig.update_xaxes(row=irow, col=icol, title = "total time [s]", 
                    linecolor = "#909497",
                    showspikes=True, # Show spike line for X-axis
                    # Format spike
                    spikethickness=2,
                    spikedash="dot",
                    spikecolor="#999999",
                    spikemode="across",)

   # --------------------
   # update layout
   fig.update_layout(
      title="<br>" +
            '<span style="font-size: 16px;"> <b>Temporals plot</b> </span>' + '<br>' +
            f'<span style="font-size: 12px;">{Path(temporal_filename).resolve()}</span>',
      legend_traceorder='grouped',
      legend_tracegroupgap = 5,
      legend_title_text='RUNS : ',
      showlegend = True,
      font = dict(color = "#909497",size=10),
      plot_bgcolor = "white", # set the background colour
      height = fig_height,
      hovermode="x", # ['x', 'y', 'closest', False, 'x unified', 'y unified']
      hoverdistance=1, #100, # Distance to show hover label of data point
      spikedistance=1000, # Distance to show spike
      margin=dict(r=20, l=20, b=50, t=100),
   )



   # --------------------
   # save file if needed
   html_filename = None
   if save_html:
      path_file = Path(temporal_filename).parent
      html_filename = Path(path_file)/(Path(temporal_filename).stem +"_estimators_plot.html")
      fig.write_html(html_filename.as_posix(), include_plotlyjs="cdn", full_html=False)
      print(f"  Html file saved in: {html_filename}")

   return html_filename.as_posix()

def post_temporal_files(filenames: list, save_pdf: bool=True, save_html: bool=True,
                        send_by_mail: bool=True, opt: str="all-in-page",print_iter_info: str=50,
                        compute_estimators: bool=False, subsampling: str=10000,stats_starting_time: str=0):
   ''' main function to post temporal file
   '''
   # --------------------
   # initialize
   start = timer.time()

   # handle error type
   if not isinstance(filenames,list):
      raise TypeError(f" filenames attribut has to be a list type")

   # --------------------
   # check if file exists and read if exists
   all_data = []
   for filename in filenames:
      if not Path(filename).is_file():
         raise FileNotFoundError(f" {filename} not found")

      # --------------------
      # parse file
      data = parse_temporals(filename,check_probe_names=False)
      # data['filename'] = filename
      all_data.append(data)
      print("  Found",len(data.keys()),"data to plot:")
      for dat in data.keys():
         print(f"   {dat}")

   # --------------------
   # save data to DataFrame
   df = rawdata_to_df(all_data)

   # --------------------
   # plot data to file
   output_files = []
   if save_pdf:
      print(f"> Plotting data to pdf. Save pdf file: {save_pdf}")
      output_filename = plot_temporal_data_in_pdf(df, filename, save_pdf, opt, print_iter_info)
      output_files.append(output_filename)
   if save_html:
      print(f"> Plotting data to html. Save html file: {save_html}")
      output_filename = plot_temporal_data_in_html(df, filename, save_html, print_iter_info)
      output_files.append(output_filename)
      if compute_estimators:
         output_filename = plot_estimators_in_html(df, filename, save_html, subsampling, stats_starting_time)
         output_files.append(output_filename)
   
   # --------------------
   # send file by email if needed
   if send_by_mail:
      if not output_files:
         print(f' > ERROR: No output file found to send by mail')
         sys.exit()

      import y2_hosttype as host
      if host.can_mail: host.mail("Temporals plot","",output_files)

   print("Done")
   end = timer.time()
   elapsed = end - start
   print(f'Total execution time : {elapsed:.2f} s')

   return df

#####################
# Main
#####################
if __name__ == '__main__':

   from docopt import docopt
   help = """
      Plot on a PDF or a HTML file all data present in the YALES2 temporal ASCII file

      Usage:
         y2_post_temporals.py (-h | --help)
         y2_post_temporals.py <file> ...
         y2_post_temporals.py <file> ... [-f <format>...] [-l <layout>] [-p <print_iter_info>] [--compute_estimators] [--subsampling <number_ite>] [--stats_starting_time <time>] [--nomail]

      Options:
         -h --help                                            Show this screen
         -f <format> --format=<format>                        Export temporals format to pdf or html file [default: html]
         -l <layout> --layout=<layout>                        Plots layouts (only for pdf): all-in-page or one-by-page [default: all-in-page]
         -p <print_iter_info> --print_info=<print_iter_info>  Print iteration information every value specified [default: 50]
         --compute_estimators                                 Compute estimators of statistics uncertainty [default: False]
         --subsampling <number_ite>                          Subsampling for computing confidence estimators : number of interpolated data [default: 10000]
         --stats_starting_time <time>                         Starting time to compute statistical estimators [default:0]
         --nomail                                             Do not send results by mail

      Examples:
         y2_post_temporals.py temporals/my_temporals.txt
         y2_post_temporals.py RUN001/temporals/my_temporals.txt RUN002/temporals/my_temporals.txt
         y2_post_temporals.py temporals/my_temporals.txt -f html -f pdf
         y2_post_temporals.py temporals/my_temporals.txt -p 10
         y2_post_temporals.py temporals/my_temporals.txt --compute_estimators --subsampling 1000
   """
   # Get arguments and load the workflow
   arguments = docopt(help)
   print(arguments)

   filenames = arguments['<file>']
   if not filenames:
      print('ERROR: bad argument ',arguments['<file>'])
      sys.exit()     

   opt = arguments['--layout']
   if not opt:
      opt="all-in-page"

   export_format = arguments['--format']
   save_pdf = False
   save_html = False
   if not export_format:
       export_format = 'html'
       save_html = True
   else:
       if 'pdfl' in export_format: save_pdf = True
       if 'html' in export_format: save_html = True

   mail = arguments['--nomail']
   if not mail:
      mail = True
   else:
      mail = False

   print_iter_info = int(arguments['--print_info'])

   compute_estimators = arguments['--compute_estimators']

   if arguments['--subsampling'] is None:
      subsampling = "1000"
   else:
      subsampling = arguments['--subsampling']

   if arguments['--stats_starting_time'] is None:
      stats_starting_time = "0"
   else:
      stats_starting_time = arguments['--stats_starting_time']

   post_temporal_files(filenames,save_pdf=save_pdf,save_html=save_html,
                      send_by_mail=mail,opt=opt,print_iter_info=print_iter_info,
                      compute_estimators=compute_estimators,subsampling=subsampling,stats_starting_time=stats_starting_time)
