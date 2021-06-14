import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil
from ublarcvapp import ublarcvapp
from MiscFunctions import cropped_np, unravel_array, reravel_array
from LArMatchModel import get_larmatch_features

def get_net_inputs_mc(PARAMS, START_ENTRY, END_ENTRY):
    # This function takes a root file path, a start entry and an end entry
    # and returns the mc track information in a form the network is
    # prepared to take as input
    print("Loading Network Inputs")
    # classes:
    # 0 = track
    # 1 = shower
    # 2 = other

    voxel = []
    uplane = []
    vplane = []
    yplane = []
    truth_ids = []
    event_ids = []
    step_dist_3d = []
    inputfilelist = ["merged_larcv_test.root","kpsreco_test.root"]
    START_ENTRY = 0
    END_ENTRY = -1
    #first get the track/shower 3d cluster points as np arrays, wire planes as np
    # outputs: list of 3d point cluster for each object, list of "truth", list of 2d planes,single 2d meta object
    full_3d_list = load_rootfile_training(inputfilelist, START_ENTRY, END_ENTRY)
    print()

    #turn 3d np arrays into voxels of 1,0

    # get 2d projections, crop around the start point

    return training_data, full_image, steps_x, steps_y, event_ids, rse_pdg_dict

def is_inside_boundaries(xt,yt,zt,buffer = 0):
    x_in = (xt <  255.999-buffer) and (xt >    0.001+buffer)
    y_in = (yt <  116.499-buffer) and (yt > -116.499+buffer)
    z_in = (zt < 1036.999-buffer) and (zt >    0.001+buffer)
    if x_in == True and y_in == True and z_in == True:
        return True
    else:
        return False

def getprojectedpixel(meta,x,y,z):

    nplanes = 3
    fracpixborder = 1.5
    row_border = fracpixborder*meta.pixel_height();
    col_border = fracpixborder*meta.pixel_width();

    img_coords = [-1,-1,-1,-1]
    tick = x/(larutil.LArProperties.GetME().DriftVelocity()*larutil.DetectorProperties.GetME().SamplingRate()*1.0e-3) + 3200.0;
    if ( tick < meta.min_y() ):
        if ( tick > meta.min_y()- row_border ):
            # below min_y-border, out of image
            img_coords[0] = meta.rows()-1 # note that tick axis and row indicies are in inverse order (same order in larcv2)
        else:
            # outside of image and border
            img_coords[0] = -1
    elif ( tick > meta.max_y() ):
        if (tick < meta.max_y()+row_border):
            # within upper border
            img_coords[0] = 0;
        else:
            # outside of image and border
            img_coords[0] = -1;

    else:
        # within the image
        img_coords[0] = meta.row( tick );


    # Columns
    # xyz = [ x, y, z ]
    xyz = array('d', [x,y,z])

    # there is a corner where the V plane wire number causes an error
    if ( (y>-117.0 and y<-116.0) and z<2.0 ):
        xyz[1] = -116.0;

    for p in range(nplanes):
        wire = larutil.Geometry.GetME().WireCoordinate( xyz, p );

        # get image coordinates
        if ( wire<meta.min_x() ):
            if ( wire>meta.min_x()-col_border ):
                # within lower border
                img_coords[p+1] = 0;
            else:
                img_coords[p+1] = -1;
        elif ( wire>=meta.max_x() ):
            if ( wire<meta.max_x()+col_border ):
                # within border
                img_coords[p+1] = meta.cols()-1
            else:
                # outside border
                img_coords[p+1] = -1
        else:
        # inside image
            img_coords[p+1] = meta.col( wire );
        # end of plane loop

    # there is a corner where the V plane wire number causes an error
    if ( y<-116.3 and z<2.0 and img_coords[1+1]==-1 ):
        img_coords[1+1] = 0;

    col = img_coords[2+1]
    row = img_coords[0]
    return col,row

def load_rootfile_training(inputfilelist, start_entry = 0, end_entry = -1):
    truthtrack_SCE = ublarcvapp.mctools.TruthTrackSCE()
    infile_larcv = inputfilelist[0]
    infile_reco = inputfilelist[1]
    iocv = larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickBackward)
    iocv.reverse_all_products() # Do I need this?
    iocv.add_in_file(infile_larcv)
    iocv.initialize()
    # ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
    # ioll.add_in_filename(infile)
    # ioll.open()

    nentries_cv = iocv.get_n_entries()

    # Get Rid of those pesky IOManager Warning Messages (orig cxx)
	# larcv::logger larcv_logger
	# larcv::msg::Level_t log_level = larcv::msg::kCRITICAL
	# larcv_logger.force_level(log_level)
    full_3d_list = []
    truthlist = []
    full_2dU_list = []
    full_2dV_list = []
    full_2dY_list = []
    runs = []
    subruns   = []
    events    = []
    entries   = []
    savedmeta = 0

    if end_entry > nentries_cv or end_entry == -1:
        end_entry = nentries_cv
    if start_entry > end_entry or start_entry < 0:
        start_entry = 0
    for i in range(start_entry, end_entry):
    # for i in range(8,9):
        print()
        print("Loading Entry:", i, "of range", start_entry, end_entry)
        print()
        iocv.read_entry(i)
        # ioll.go_to(i)

        ev_wire    = iocv.get_data(larcv.kProductImage2D,"wire")
        # Get Wire ADC Image to a Numpy Array
        img_v = ev_wire.Image2DArray()
        u_wire_image2d = img_v[0]
        v_wire_image2d = img_v[1]
        y_wire_image2d = img_v[2]
        if i==0:
            savedmeta = y_wire_image2d.meta()

        rows = y_wire_image2d.meta().rows()
        cols = y_wire_image2d.meta().cols()

        u_wire_np = larcv.as_ndarray(u_wire_image2d)
        v_wire_np = larcv.as_ndarray(v_wire_image2d)
        y_wire_np = larcv.as_ndarray(y_wire_image2d)

        # change to append for every reco object
        full_2dU_list.append(u_wire_np)
        full_2dV_list.append(v_wire_np)
        full_2dY_list.append(y_wire_np)
        runs.append(ev_wire.run())
        subruns.append(ev_wire.subrun())
        events.append(ev_wire.event())
        entries.append(i)

        print("SHAPE TEST")
        print(y_wire_np.shape)


    return full_image_list, ev_trk_xpt_list, ev_trk_ypt_list, runs, subruns, events, filepaths, entries, track_pdgs
