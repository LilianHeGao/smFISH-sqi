import argparse
import numpy as np
from skimage.feature import blob_log

def read_im(path,return_pos=False):
    import zarr,os
    from dask import array as da
    dirname = os.path.dirname(path)
    fov = os.path.basename(path).split('_')[-1].split('.')[0]
    #print("Bogdan path:",path)
    file_ = dirname+os.sep+fov+os.sep+'data'
    #image = zarr.load(file_)[1:]
    image = da.from_zarr(file_)[1:]

    shape = image.shape
    #nchannels = 4
    xml_file = os.path.dirname(path)+os.sep+os.path.basename(path).split('.')[0]+'.xml'
    if os.path.exists(xml_file):
        txt = open(xml_file,'r').read()
        tag = '<z_offsets type="string">'
        zstack = txt.split(tag)[-1].split('</')[0]
        
        tag = '<stage_position type="custom">'
        x,y = eval(txt.split(tag)[-1].split('</')[0])
        
        nchannels = int(zstack.split(':')[-1])
        nzs = (shape[0]//nchannels)*nchannels
        image = image[:nzs].reshape([shape[0]//nchannels,nchannels,shape[-2],shape[-1]])
        image = image.swapaxes(0,1)
    shape = image.shape
    if return_pos:
        return image,x,y
    return image
	
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # load image: (C, Z, Y, X) or (C, Y, X)
    im = read_im(args.zarr)

    # DAPI is last channel
    spot_channels = im[:-1]

    spots = []

    for c in range(spot_channels.shape[0]):
        img = np.max(spot_channels[c], axis=0) if spot_channels[c].ndim == 3 else spot_channels[c]

        blobs = blob_log(
            img,
            min_sigma=1.0,
            max_sigma=2.5,
            num_sigma=5,
            threshold=0.02,
        )

        # blobs: (row, col, sigma)
        if blobs.size > 0:
            spots.append(blobs[:, :2])

    if spots:
        spots_rc = np.vstack(spots).astype(np.float32)
    else:
        spots_rc = np.zeros((0, 2), dtype=np.float32)

    np.save(args.out, spots_rc)
    print("Saved spots:", spots_rc.shape, "->", args.out)


if __name__ == "__main__":
    main()
