import ROOT
import math
import numpy as np
from larcv import larcv
from larflow import larflow
from larlite import larlite
from array import *
from larlite import larutil
from ublarcvapp import ublarcvapp
import matplotlib.pyplot as plt
from scipy import asarray as ar,exp



def get_net_inputs_mc(START_ENTRY, END_ENTRY):
    # This function takes a root file path, a start entry and an end entry
    # and returns the mc track information in a form the network is
    # prepared to take as input
    print("Loading Network Inputs")
    # classes:
    # 0 = track
    # 1 = shower
    # 2 = other


    inputfilelist = ["../model/merged_larcv_test.root","../model/kpsreco_test.root"]
    #first get the track/shower 3d cluster points as np arrays, wire planes as np
    # outputs: list of tuple for each entry
    # for each entry:voxel,u,v,y,class,qual,r,s,e
    trainingdata = load_rootfile_training(inputfilelist, False, START_ENTRY, END_ENTRY)
    print()

    return trainingdata
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

    return img_coords

def load_rootfile_training(inputfilelist, SAVEPLOTS = False, start_entry = 0, end_entry = -1):
    infile_larcv = inputfilelist[0]
    infile_reco = inputfilelist[1]
    iocv = larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickBackward)
    iocv.reverse_all_products() # Do I need this?
    iocv.add_in_file(infile_larcv)
    iocv.initialize()
    ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
    ioll.add_in_filename(infile_larcv)
    ioll.open()

    # get the reco info
    kps_f = ROOT.TFile(infile_reco,"read")
    in_tree = kps_f.Get("KPSRecoManagerTree")

    nentries_cv = iocv.get_n_entries()
    training_data_v = []

    # classes:
    # 0 = electron
    # 1 = gamma
    # 2 = proton
    # 3 = muon
    # 4 = charged pi0
    # 5 = other

    if end_entry > nentries_cv or end_entry == -1:
        end_entry = nentries_cv
    if start_entry > end_entry or start_entry < 0:
        start_entry = 0

    # testing size:=
    feattrue_shower=[]
    feattrue_track=[]

    for ientry in range(start_entry, end_entry):
    # for i in range(8,9):
        print()
        print("Loading Entry:", ientry, "of range", start_entry, end_entry)
        print()
        iocv.read_entry(ientry)
        in_tree.GetEntry(ientry)
        ioll.go_to(ientry)

        # get branches
        pnu_sel_v = in_tree.nu_sel_v
        nufitted_v = in_tree.nufitted_v

        ev_wire    = iocv.get_data(larcv.kProductImage2D,"wire")
        # Get Wire ADC Image to a Numpy Array
        img_v = ev_wire.Image2DArray()
        u_wire_image2d = img_v[0]
        v_wire_image2d = img_v[1]
        y_wire_image2d = img_v[2]

        # load objects for truth matching
        ev_mcshower = ioll.get_data(larlite.data.kMCShower,"mcreco");
        ev_mctrack = ioll.get_data(larlite.data.kMCTrack,"mcreco");

        # loop through all reco vertices
        nvtx = pnu_sel_v.size()
        for ivtx in range(nvtx):
            # print("Vertex: ",ivtx)
            # load in the reco track/shower 3d points.
            nufit_shower_v = nufitted_v[ivtx].shower_v
            nufit_showertrunk_v = nufitted_v[ivtx].shower_trunk_v
            nufit_showerpca_v = nufitted_v[ivtx].shower_pcaxis_v

            nufit_track_v = nufitted_v[ivtx].track_hitcluster_v
            nufit_track_larlite_v = nufitted_v[ivtx].track_v
            # only want to look at well reco'd vertex objects
            if pnu_sel_v[ivtx].dist2truevtx < 2:

                if nufit_shower_v.size() >0:
                # if False: #false to skip to focus on tracks
                    for ishower in range(nufit_shower_v.size()):

                        # make into larflow cluster_t object
                        nufit_shower_c = larflow.reco.cluster_from_larflowcluster(nufit_shower_v[ishower])
                        shower_points = nufit_shower_c.points_v
                        voxel = get3dvoxel(shower_points, nufitted_v[ivtx])

                        matchtrue, feattrue = showertruthmatching(nufit_shower_v[ishower],nufit_showertrunk_v[ishower],
                                        nufit_showerpca_v[ishower], img_v , ev_mcshower)
                        type = -1
                        if abs(matchtrue) == 11:
                            type = 0
                        elif abs(matchtrue) == 22:
                            type = 1
                        else:
                            type =5
                            print("other",matchtrue)

                        type_v=[0,0,0,0,0,0]
                        type_v[type] = 1


                        # get 2d projection
                        proj = get2dprojection(nufit_shower_c,nufitted_v[ivtx],img_v)

                        if SAVEPLOTS:
                            plotvoxel(voxel,ientry,type,feattrue,proj)

                        x = (voxel,proj[0],proj[1],proj[2],
                            type_v,feattrue,ev_wire.run(),ev_wire.subrun(),ev_wire.event())
                        training_data_v.append(x)


                if nufit_track_v.size() >0:
                    for itrack in range(nufit_track_v.size()):
                        # make into larflow cluster_t object
                        nufit_track_c = larflow.reco.cluster_from_larflowcluster(nufit_track_v[itrack])
                        track_points = nufit_track_c.points_v
                        voxel = get3dvoxel(track_points, nufitted_v[ivtx])
                        # truth matching
                        matchtrue, feattrue = tracktruthmatching(nufit_track_larlite_v[itrack], ev_mctrack)
                        type = -1
                        if abs(matchtrue) == 2212:
                            type = 2
                        elif abs(matchtrue) == 13:
                            type = 3
                        elif abs(matchtrue) == 211:
                            type = 4
                        else:
                            type = 5
                            print("other",matchtrue)
                        type_v=[0,0,0,0,0,0]
                        type_v[type] = 1


                        proj = get2dprojection(nufit_track_c,nufitted_v[ivtx],img_v)
                        if SAVEPLOTS:
                            plotvoxel(voxel,ientry,type,feattrue,proj)

                        x = (voxel,proj[0],proj[1],proj[2],
                            type_v,feattrue,ev_wire.run(),ev_wire.subrun(),ev_wire.event())
                        training_data_v.append(x)


    return training_data_v

def get3dvoxel(reco3d,vtx,size=1024,scale=3):
    # make empty voxel to fill in:
    voxel = np.zeros((size,size,size))
    voxel = np.zeros((size,size,size))
    # fill vertex
    voxel[math.floor(size/2)][math.floor(size/2)][math.floor(size/2)]=1
    # only for plotting fill corners
    voxel[0][0][0]=1
    voxel[size-1][size-1][size-1]=1

    for i in range(len(reco3d)):
        x = reco3d[i][0]
        y = reco3d[i][1]
        z = reco3d[i][2]
        # voxel coordinates will each be .25 cm
        deltax = math.floor((x-vtx.pos[0])*scale)
        deltay = math.floor((y-vtx.pos[1])*scale)
        deltaz = math.floor((z-vtx.pos[2])*scale)
        if(abs(deltax)<(size/2) and abs(deltay)<(size/2)and abs(deltaz)<(size/2) ):
            voxel[math.floor(size/2)+deltax][math.floor(size/2)+deltay][math.floor(size/2)+deltaz] =1

    return voxel

def plotvoxel(voxel,ientry,classtype,featdist,proj):
    if (False):
        # prepare some coordinates
        x=[]
        y=[]
        z=[]

        for i in range(len(voxel)):
            print(i, len(voxel))
            for j in range(len(voxel[i])):
                for k in range(len(voxel[i][j])):
                    if voxel[i][j][k]==1:
                        x.append(i)
                        y.append(j)
                        z.append(k)
        # and plot everything
        # print("finished plotloop")
        type = gettypestring(classtype)
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(x,y,z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(type+" "+str(featdist))
        plt.savefig("voxeltest_"+str(ientry)+"_"+type+"_"+str(featdist)+".png")
        plt.close()

    if (False):
        for i in range(3):
            fig = plt.figure(figsize=(6, 3.2))
            ax = fig.add_subplot(111)
            proj[i][proj[i] > 100] =100
            plt.imshow(proj[i])
            plt.gca().invert_yaxis()
            ax.set_aspect('equal')
            cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
            cax.get_xaxis().set_visible(False)
            cax.get_yaxis().set_visible(False)
            cax.patch.set_alpha(0)
            cax.set_frame_on(False)
            plt.colorbar(orientation='vertical')
            type = gettypestring(classtype)
            ax.set_title(type+" "+str(featdist)+" plane:"+str(i))
            plt.savefig("projtest_"+str(ientry)+"_"+str(i)+"_"+type+"_"+str(featdist)+".png")
            plt.close()



    return

def get2dprojection(reco_points,vtx,img_v):
    # output is a a list of 2d projections
    size =512
    proj_v = np.zeros((3,size,size))
    # loop over planes
    # get 2d projection of vertex:
    vtx2d = getprojectedpixel(img_v.at(0).meta(),vtx.pos[0],vtx.pos[1],vtx.pos[2])

    foundpts=[0,0,0]
    for ihit in range(len(reco_points.points_v)):
        hit2d = getprojectedpixel(img_v.at(0).meta(),reco_points.points_v[ihit][0],reco_points.points_v[ihit][1],reco_points.points_v[ihit][2])
        for p in range(3):
            meta = img_v.at(p).meta()
            vtx_row = vtx2d[0]
            vtx_col = vtx2d[p+1]
            row = hit2d[0]
            col = hit2d[p+1]

            row_crop = math.floor(size/2) + (row-vtx_row)
            col_crop = math.floor(size/2) + (col-vtx_col)

            if  (row_crop<size and col_crop<size and proj_v[p][row_crop][col_crop] == 0 and img_v[p].pixel(row,col)>10):
                proj_v[p][row_crop][col_crop] = img_v[p].pixel(row,col)
                foundpts[p]+=1

    return proj_v

def gettypestring(classtype):
    # classes:
    # 0 = electron
    # 1 = gamma
    # 2 = proton
    # 3 = muon
    # 4 = charged pi0
    # 5 = other
    if classtype == 0:
        type = "electron"
    elif classtype == 1:
        type = "gamma"
    elif classtype == 2:
        type = "proton"
    elif classtype == 3:
        type = "muon"
    elif classtype == 4:
        type = "charged pion"
    else:
        type = "other"
    return type

def showertruthmatching(shower_v, trunk_v, pca_v, img_v , mcshower_v):

    _psce = larutil.SpaceChargeMicroBooNE()

    meta = img_v.front().meta();


    start_pos = [float(trunk_v.LocationAtPoint(0)[0]),
				 float(trunk_v.LocationAtPoint(0)[1]),
				 float(trunk_v.LocationAtPoint(0)[2])]
    end_pos = [float(trunk_v.LocationAtPoint(1)[0]),
				 float(trunk_v.LocationAtPoint(1)[1]),
				 float(trunk_v.LocationAtPoint(1)[2])]

    dist = 0.0
    dir = [0.0,0.0,0.0]
    for i in range(3):
      dir[i] = (end_pos[i]-start_pos[i])
      dist += dir[i]*dir[i]
    dist = math.sqrt(dist)
    if dist>0 :
      for i in range(3):
        dir[i] =dir[i]/float(dist)

    start_tick = start_pos[0]/larutil.LArProperties.GetME().DriftVelocity()/0.5+3200
    if ( start_tick<meta.min_y() or start_tick>meta.max_y() ):
        #try to shorten trunk to stay in image
        fix = False
        if ( start_tick<=meta.min_y() and dir[0]!=0.0) :
        	mintick = (meta.pos_y( 1 )-3200)*0.5*larutil.LArProperties.GetME().DriftVelocity()
        	s = (mintick-start_pos[0])/dir[0]
        	fix = True
        	for i in range(3):
        	   start_pos[i] = start_pos[0] + s*dir[i]
        	start_tick = meta.pos_y(1)


        elif ( start_tick>=meta.max_y() and dir[0]!=0.0):
        	maxtick = (meta.pos_y( int(meta.rows())-1 )-3200)*0.5*larutil.LArProperties.GetME().DriftVelocity()
        	s = (maxtick-start_pos[0])/dir[0];
        	fix = True
        	for i in range(3):
        	  start_pos[i] = start_pos[0] + s*dir[i]
        	start_tick = meta.pos_y( int(meta.rows())-1 )

    start_row = float(meta.row(start_tick))

    end_tick = end_pos[0]/larutil.LArProperties.GetME().DriftVelocity()/0.5+3200
    if ( end_tick<=meta.min_y() or end_tick>=meta.max_y() ):
        # try to shorten trunk to stay in image
        fix = False
        s = 0.0
        if ( end_tick<=meta.min_y() and dir[0]!=0.0):
            mintick = (meta.pos_y( 1 )-3200)*0.5*larutil.LArProperties.GetME().DriftVelocity()
            s = (mintick-start_pos[0])/dir[0]
            fix = True
            for i in range(3):
                end_pos[i] = start_pos[0] + s*dir[i]
            end_tick = meta.pos_y(1)

        elif ( end_tick>=meta.max_y() and dir[0]!=0.0):
            maxtick = (meta.pos_y( int(meta.rows())-1 )-3200)*0.5*larutil.LArProperties.GetME().DriftVelocity()
            s = (maxtick-start_pos[0])/dir[0]
            fix = True
            for i in range(3):
              end_pos[i] = start_pos[0] + s*dir[i]
            end_tick = meta.pos_y( int(meta.rows())-1 )

    end_row = float(meta.row(end_tick))
    dist = 0.0
    for i in range(3):
        dir[i] = end_pos[i]-start_pos[i]
        dist += dir[i]*dir[i]

    dist = math.sqrt(dist)
    for i in range(3):
        dir[i] /= dist;

    # initialize values
    min_index = -1
    min_feat_dist = 1e9
    vertex_err_dist = 0
    match_pdg = 0
    dir_cos = 0.0

    # start loop over showers
    for  ishower in range(mcshower_v.size()):

        mcshower = mcshower_v[ishower]

        if ( mcshower.Origin()!=1 ):
            continue # not neutrino origin

        profile = mcshower.DetProfile()

        # default shower_dir: from the truth
        shower_dir = mcshower.Start().Momentum().Vect()
        pmom = shower_dir.Mag()
        mcstart = mcshower.Start().Position().Vect()
        pstart  = profile.Position().Vect() # start of shower profile

        if ( mcshower.PdgCode()==22 ) :
            #if gamma, we want to use the dir and start from the profile
            shower_dir = profile.Momentum().Vect()
            pmom = shower_dir.Mag()
            mcstart = pstart

        # copy TVector3 to vector<float>, so we can use geofunc
        mcdir = [0,0,0]
        fmcstart = [0,0,0]
        fmcend = [0,0,0]
        for i in range(3):
            shower_dir[i] /= pmom
            mcdir[i] = float(shower_dir[i])
            fmcstart[i] = mcstart[i]
            fmcend[i] = fmcstart[i] + 10.0*mcdir[i]

        if ( mcshower.PdgCode()==22 ):
            # // space charge correction
            s_offset = _psce.GetPosOffsets(mcstart[0],mcstart[1],mcstart[2])
            fmcstart[0] = fmcstart[0] - s_offset[0] + 0.7
            fmcstart[1] = fmcstart[1] + s_offset[1]
            fmcstart[2] = fmcstart[2] + s_offset[2]

            e_offset = _psce.GetPosOffsets(fmcend[0],fmcend[1],fmcend[2])
            fmcend[0] = fmcend[0] - e_offset[0] + 0.7
            fmcend[1] = fmcend[1] + e_offset[1]
            fmcend[2] = fmcend[2] + e_offset[2]

        fsce_dir = [0,0,0]
        sce_dir_len = 0.0
        for i in range(3):
            fsce_dir[i] = fmcend[i] - fmcstart[i]
            sce_dir_len += (fsce_dir[i]*fsce_dir[i])

        sce_dir_len = math.sqrt( sce_dir_len )
        if ( sce_dir_len>0 ):
            for i in range(3):
                fsce_dir[i] /= sce_dir_len

        # finally!
        Tlen = 0.0
        Tproj = 0.0
        for v in range(3):
            Tlen += dir[v]*dir[v];
            Tproj += dir[v]*( fmcstart[v]-start_pos[v] );

        Tlen = math.sqrt(Tlen);
        Tproj=Tproj/float(Tlen)

        dvertex = Tproj
        fcos = 0.0
        for i in range(3):
            fcos += fsce_dir[i]*dir[i]

        goodmetric = (1.0-fcos)*(1.0-fcos) + (dvertex*dvertex/9.0) # dvertex has a sigma of 3 cm
        if ( min_feat_dist>goodmetric ):
            dir_cos = fcos
            vertex_err_dist = dvertex
            min_feat_dist = goodmetric
            match_pdg = mcshower.PdgCode()
            min_index =  ishower

    #end of loop over showers
    # print("Best true shower match: ")
    # print(" - feat_dist=", min_feat_dist)
    # print(" - vertex_dist=",vertex_err_dist)
    # print(" - true-dir-cos=",dir_cos)
    # print(" - match PDG code=", match_pdg)
    # print(" - true min index=", min_index)
    min_feat_dist = 1.0 - (2.0/3.14159)*math.atan(min_feat_dist)

    return match_pdg,min_feat_dist

def tracktruthmatching(track_v, mctrack_v):

    # initialize SCE object
    _psce = larutil.SpaceChargeMicroBooNE()
    # make list of true start+end
    # make list of true direction
    true_start_v = []
    true_end_v = []
    true_dir_v = []
    true_PDG_v = []
    true_usable_v = []
    for itrack in range(len(mctrack_v)):
        mctrack = mctrack_v[itrack]
        if ( mctrack.Origin()!=1 or mctrack.PdgCode() >3000 or mctrack.PdgCode()==2112 ):
            continue # not neutrino origin

        start_root = mctrack.Start().Position().Vect()
        end_root = mctrack.End().Position().Vect()
        dir_root = mctrack.Start().Momentum().Vect()

        mcdir = [0,0,0]
        mcstart = [0,0,0]
        mcend = [0,0,0]
        distmc = 0
        for i in range(3):
            mcdir[i] = float(dir_root[i])
            mcstart[i] = float(start_root[i])
            mcend[i] = float(end_root[i])
            testdir = (end_root[i]-start_root[i])
            distmc += testdir*testdir
        distmc = math.sqrt(distmc)

        fmcstart = [0,0,0]
        fmcend = [0,0,0]

        s_offset = _psce.GetPosOffsets(mcstart[0],mcstart[1],mcstart[2])
        fmcstart[0] = mcstart[0] - s_offset[0] + 0.7
        fmcstart[1] = mcstart[1] + s_offset[1]
        fmcstart[2] = mcstart[2] + s_offset[2]

        e_offset = _psce.GetPosOffsets(fmcend[0],fmcend[1],fmcend[2])
        fmcend[0] = mcend[0] - e_offset[0] + 0.7
        fmcend[1] = mcend[1] + e_offset[1]
        fmcend[2] = mcend[2] + e_offset[2]

        fsce_dir = [0,0,0]
        sce_dir_len = 0.0
        for i in range(3):
            fsce_dir[i] = fmcend[i] - fmcstart[i]
            sce_dir_len += (fsce_dir[i]*fsce_dir[i])

        sce_dir_len = math.sqrt( sce_dir_len )
        if ( sce_dir_len>0 ):
            for i in range(3):
                fsce_dir[i] /= sce_dir_len


        if distmc>0:
            true_start_v.append(fmcstart)
            true_end_v.append(fmcend)
            true_dir_v.append(fsce_dir)
            true_PDG_v.append(mctrack.PdgCode())


        use = 0
        startin = is_inside_boundaries(mcstart[0],mcstart[1],mcstart[2],0)
        endin = is_inside_boundaries(mcend[0],mcend[1],mcend[2],0)
        if (endin==False and startin==False):
            use =3
        elif(endin==False):
            use = 1
        elif(startin==False):
            use = 2
        true_usable_v.append(use)
    #end of loop over true

    # get reco variables
    start_pos = [float(track_v.LocationAtPoint(0)[0]),
				 float(track_v.LocationAtPoint(0)[1]),
				 float(track_v.LocationAtPoint(0)[2])]
    tracklen = track_v.NumberTrajectoryPoints()
    end_pos = [float(track_v.LocationAtPoint(tracklen-1)[0]),
				 float(track_v.LocationAtPoint(tracklen-1)[1]),
				 float(track_v.LocationAtPoint(tracklen-1)[2])]

    # get reco start
    # get reco end
    # get reco dir
    dist = 0.0
    dir = [0.0,0.0,0.0]
    for i in range(3):
      dir[i] = (end_pos[i]-start_pos[i])
      dist += dir[i]*dir[i]
    dist = math.sqrt(dist)
    if dist>0 :
      for i in range(3):
        dir[i] =dir[i]/float(dist)


    # initialize values
    min_index = -1
    min_feat_dist = 1e9
    vertex_err_dist = 0
    match_pdg = 0
    dir_cos = 0.0
    for itrack in range(len(true_start_v)):
        dist_start = 0.0
        dist_start_inv = 0.0
        for v in range(3):
            dist_start += (true_start_v[itrack][v]-start_pos[v])**2
            dist_start_inv += (true_start_v[itrack][v]-end_pos[v])**2
        dist_start = math.sqrt(dist_start)
        dist_start_inv = math.sqrt(dist_start_inv)
        dvertex = min(dist_start,dist_start_inv)

        fcos = 0.0
        for i in range(3):
            # print(true_dir_v[itrack][i])
            fcos +=  true_dir_v[itrack][i]*dir[i]


        goodmetric = math.sqrt((1.0-abs(fcos))*(1.0-abs(fcos)) +(dvertex*dvertex)/25.0) # dvertex has a sigma of 3 cm
        if ( min_feat_dist>goodmetric ):
            dir_cos = fcos
            vertex_err_dist = dvertex
            min_feat_dist = goodmetric
            match_pdg = true_PDG_v[itrack]
            min_index =  itrack

    # return pdg of best match
    # print("Best true shower match: ")
    # print(" - feat_dist=", min_feat_dist)
    # print(" - start_dist=",vertex_err_dist)
    # print(" - true-dir-cos=",dir_cos)
    # print(" - match PDG code=", match_pdg)
    # print(" - true min index=", min_index)
    min_feat_dist = 1.0 - (2/3.14159)*math.atan(min_feat_dist)


    return match_pdg, min_feat_dist
