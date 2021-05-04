# from __future__ import annotations


from PIL import Image, ImageDraw, ImageFont

import functools
import glob
import itertools
import typing
import collections
import contextlib
import os
import multiprocessing
import pathlib

size_4k = (3840, 2160)
size_8k = (7680, 4320)

thumb_size = size_4k  # Either, say, size_4k or None

class CropDims(typing.NamedTuple):
    left: int
    upper: int
    right: int
    lower: int

crop_top_f_fine = CropDims(872, 178, 872+2094, 178+561)
crop_top_bit = CropDims(0, 100, 3840, 100 + 2160//2)
crop_top_mid = CropDims(872, 100, 872+2094, 100 + 2160//2)
crop_dims: typing.Optional[CropDims] = None


class SimulationInfo(typing.NamedTuple):
    throttler_relaxation: float
    # adj_strain_ratio: float
    adj_strain_ratio_true: float
    n_steps_minor_max: int
    source_file_name: str
    # scaling: str
    freedom_cases: str
    scale_model_x: float
    scale_model_y: float
    perturbator: str


    @classmethod
    def from_meta_txt(cls, fn: str) -> "SimulationInfo":
        def line_start(l):
            working = l.strip().split()
            if working:
                return working[0]

        working_dict = {}

        _NEW_VARIABLES = {
            "freedom_cases": "bending_pure",
            "scale_model_x": 1.0,
            "scale_model_y": 1.0,
            "perturbator": "None",
        }

        working_dict.update(_NEW_VARIABLES)

        # Backwards compatible with the old adj_strain_ratio bug...
        full_annotations = cls.__annotations__.copy()
        full_annotations["adj_strain_ratio"] = full_annotations["adj_strain_ratio_true"]
        with open(fn) as f_in:
            lines_stripped = (l.strip() for l in f_in)
            lines_to_consider = list(l for l in lines_stripped if line_start(l) in full_annotations)
            for l in lines_to_consider:
                key, val = l.split(maxsplit=1)
                val_as_native_type = (full_annotations[key])(val) 
                working_dict[key] = val_as_native_type

        if "adj_strain_ratio" in working_dict and not "adj_strain_ratio_true" in working_dict:
            working_dict["adj_strain_ratio_true"] = working_dict["adj_strain_ratio"] ** 2

        _ = working_dict.pop("adj_strain_ratio", None)
        return cls(**working_dict)


class SubFrame(typing.NamedTuple):
    sim_info: SimulationInfo
    fn: typing.Optional[str]


class Job(typing.NamedTuple):
    sub_frames: typing.List[SubFrame]
    fn_out: str


def make_distinctive_string(this_sim_info: SimulationInfo, all_sim_infos: typing.List[SimulationInfo]) -> str:
    """Find the things which are distinctive and return that in a string"""
    
    # See which ones there are more than one of
    all_vals = collections.defaultdict(set)
    for sim_info in all_sim_infos:
        for key, val in sim_info._asdict().items():
            all_vals[key].add(val)

    not_all_same = {key for key, val_set in all_vals.items() if len(val_set) > 1}

    def nice_val(val) -> str:
        if isinstance(val, float):
            return f'{val:.5g}'

        return str(val)

    bits = [f"{key}={nice_val(val)}" for key, val in this_sim_info._asdict().items() if key in not_all_same]
    return "\n".join(bits)


def test_distinctive_strings():
    sim_info_main = SimulationInfo.from_meta_txt(r"E:\Simulations\CeramicBands\v7\pics\70\Meta.txt")
    sim_info_other = SimulationInfo.from_meta_txt(r"E:\Simulations\CeramicBands\v7\pics\71\Meta.txt")

    distinctive_string = make_distinctive_string(sim_info_main, [sim_info_main, sim_info_other])
    assert distinctive_string == "adj_strain_ratio=0.2"


@functools.lru_cache()
def _tile_dimensions(dim_single: typing.Tuple[int, int], n: int) -> int:
    """See how many tiled images we need. For example, 4 images fit in 2x2. Seven images need 3x3. 
    If thumbnails have a different aspect ratio to the output it might not be square (e.g., 2x3)"""

    def would_fit(nxy):
        return nxy[0] * nxy[1] >= n

    def wasted_space(nxy):
        nx, ny = nxy
        proposed_canvas_used = (dim_single[0] * nx, dim_single[1] * ny)
        aspect_4k = size_4k[0] / size_4k[1]
        aspect_proposal = proposed_canvas_used[0] / proposed_canvas_used[1]

        ratio_of_ratios = aspect_proposal / aspect_4k

        if ratio_of_ratios > 1:
            # Too wide - will need to pad vertical
            proposed_canvas_padded = (proposed_canvas_used[0], proposed_canvas_used[1] * ratio_of_ratios)

        elif ratio_of_ratios < 1:
            # Too tall - will need to pad horizontally.
            proposed_canvas_padded = (proposed_canvas_used[0] / ratio_of_ratios, proposed_canvas_used[1])

        else:
            # Bingo!
            proposed_canvas_padded = proposed_canvas_used

        used_space_ratio = (dim_single[0] * dim_single[1] * n) / (proposed_canvas_padded[0] * proposed_canvas_padded[1])
        return 1.0 - used_space_ratio

    max_considered_tiles = n
    nxy_all = itertools.product(range(1, max_considered_tiles+1), range(1, max_considered_tiles+1))
    nxy_fit = ((wasted_space(nxy), nxy) for nxy in nxy_all if would_fit(nxy))
    least_wasted_space = sorted(nxy_fit)
    return least_wasted_space[0][1]



T_Num = typing.Union[int, float]
def _add_text_centred_at(image: Image, xy_pos: typing.Tuple[T_Num, T_Num], text: str):
    font = ImageFont.truetype("pala.ttf", 48)

    # Position the text
    text_size = font.getsize(text)

    centred_pos = (int(xy_pos[0]-0.5*text_size[0]), int(xy_pos[1]-0.5*text_size[1]) )
    draw = ImageDraw.Draw(image)
    draw.text(centred_pos, text, font=font)




def compose_images(job: Job):

    all_sim_infos = [sub_frame.sim_info for sub_frame in job.sub_frames]

    with contextlib.ExitStack() as exit_stack:
        images = {}
        for sub_frame in job.sub_frames:
            if sub_frame.fn:
                this_image_file = exit_stack.enter_context(Image.open(sub_frame.fn))
                images[sub_frame] = this_image_file

        dims, modes = set(), set()
        for image in images.values():
            dims.add(image.size)
            modes.add(image.mode)

        if len(dims) != 1:
            raise ValueError(f"Expecting to get a single image dimension. Got {dims}.")

        if len(modes) != 1:
            raise ValueError(f"Expecting a single mode - got {modes}")

        # Compose the images
        dim_single_full = dims.pop()

        if crop_dims:
            dim_single_x = crop_dims.right - crop_dims.left
            dim_single_y = crop_dims.lower - crop_dims.upper
            assert dim_single_x <= dim_single_full[0]
            assert dim_single_y <= dim_single_full[1]
            dim_single = (dim_single_x, dim_single_y)

        else:
            dim_single = dim_single_full 
        
        mode = modes.pop()
        n_tiles_x, n_tiles_y = _tile_dimensions(dim_single, len(job.sub_frames))
        dim = (dim_single[0]*n_tiles_x, dim_single[1]*n_tiles_y)
        big_image = exit_stack.enter_context(Image.new(mode, dim))

        for idx, sub_frame in enumerate(job.sub_frames):
            tile_top, tile_left = divmod(idx, n_tiles_x)
            tile_bottom = n_tiles_y - tile_top - 1

            x_offset = tile_left * dim_single[0]
            y_offset = tile_bottom * dim_single[1]

            if sub_frame.fn:
                image = images[sub_frame]
                if crop_dims:
                    image_cropped = image.crop(crop_dims)

                else:
                    image_cropped = image
                big_image.paste(image_cropped, (x_offset,y_offset))

            # Pop in a caption
            caption = make_distinctive_string(sub_frame.sim_info, all_sim_infos)
            xy_pos = (x_offset + 0.5*dim_single[0], y_offset + 0.9 * dim_single[1])
            _add_text_centred_at(big_image, xy_pos, caption)


        # Don't need it that big
        if thumb_size:
            big_image.thumbnail(thumb_size, Image.ANTIALIAS)

        # Make directory if needed.
        out_dir = os.path.split(job.fn_out)[0]
        os.makedirs(out_dir, exist_ok=True)

        big_image.save(job.fn_out)

    return job.fn_out


class ImageProducer():
    sim_info: SimulationInfo
    d: str
    rel_files: typing.List[str]

    def __init__(self, d: str):
        self.d = d
        self.sim_info = SimulationInfo.from_meta_txt(os.path.join(self.d, "meta.txt"))

        with os.scandir(self.d) as it:
            rel_files = [entry.path for entry in it if entry.name.startswith("Case-")]

        self.rel_files = sorted(rel_files)

    def __iter__(self):
        # Don't include the final file in case it's being worked on or modified.
        return iter(self.rel_files[:-1])


def interleave_directories(dirs: typing.List[str], out_dir: str) -> typing.Iterable[Job]:
    """Zip together the frames, and have it all sorted"""

    image_producers = [ImageProducer(d) for d in dirs]

    def sort_key(image_producer: ImageProducer):
        return image_producer.sim_info

    image_producers.sort(key=sort_key)
    for imp in image_producers:
        print(imp.sim_info, len(imp.rel_files), imp.d)

    for global_frame_idx, fns_or_nones in enumerate(itertools.zip_longest(*image_producers)):
        sub_frames = []
        for image_producer, fn_or_none in zip(image_producers, fns_or_nones):
            sub_frame = SubFrame(sim_info=image_producer.sim_info, fn=fn_or_none)
            sub_frames.append(sub_frame)

        fn_out = os.path.join(out_dir, f"CaseComb-{global_frame_idx:04}.png")
        yield Job(sub_frames=sub_frames, fn_out=fn_out)


def test_compose_image():
    end_dirs = ['70', '71', '72', '73', '74', '75']
    dirs = [os.path.join(r"E:\Simulations\CeramicBands\v7\pics", ed) for ed in end_dirs]

    sub_frames = []
    for d in dirs:
        sub_frame = SubFrame( sim_info=SimulationInfo.from_meta_txt(os.path.join(d, "meta.txt")), fn=os.path.join(d, "Case-0066.png"))        
        sub_frames.append(sub_frame)

    fn_big_test = r"c:\temp\img\composed_image.png"
    sub_frames.sort()

    # None frame means blank image
    temp_frame = sub_frames[1]
    temp_frame = temp_frame._replace(fn=None)
    sub_frames[1] = temp_frame
    job = Job(sub_frames=sub_frames, fn_out=fn_big_test)
    compose_images(job)

    # Make sure we can open it.
    with Image.open(fn_big_test) as f_image:
        assert f_image.size

def test_interleave():
    end_dirs_round2 = ['6K', '6Q', '6R', '6S', '6T', '6U']
    dirs = [os.path.join(r"E:\Simulations\CeramicBands\v7\pics", ed) for ed in end_dirs_round2]
    for job in interleave_directories(dirs, r"c:\temp\img"):
        pass


def make_images():
    end_dirs = ['70', '71', '72', '73', '74', '75', '76', '77', '78']
    dirs = [os.path.join(r"E:\Simulations\CeramicBands\v7\pics", ed) for ed in end_dirs]

    for job in interleave_directories(dirs, r"c:\temp\img"):
        print(job.fn_out)
        compose_images(job)


def do_all_multi_process():
    N_WORKERS=14
    with multiprocessing.Pool(N_WORKERS) as pool:
        end_dirs_round1 = ['70', '71', '72', '73', '74', '75', '76', '77', '78']
        end_dirs_round2 = ['6R', '6S', '6T', '6U']
        end_dirs_round3 = ['6Y', '79', '7A', '7B']
        end_dirs_round4 = ['6Y', '79', '7A', '7B', '7C', '7D', '7E', '7F',]
        end_dirs_adj_low = ['7C', '79', '7D', '7A']
        end_dirs_adj_high = ['7E', '7B', '7F', '6Y']
        
        end_dirs_low2 = [ '79', '7D', '7A']

        pure_med = ['8S', '8W', '90', '94']
        threep_med = ['8U', '8Y', '92', '96']

        pure_fine = ['8T', '8X', '91', '95']
        threep_fine = ['8V', '8Z', '93', '97']

        pure_fine_narrow = ['8T', '8X']

        pure_fine_low = ['98', '99', '9A', '9B']
        pure_fine_med = ['9C', '9D', '9E', '9F']

        pure_fine_all = ['98', '99', '9A', '9B', '9C', '9D', '9E', '9F']

        def do_one(out_dir, end_dirs):
            dirs = [os.path.join(r"E:\Simulations\CeramicBands\v7\pics", ed) for ed in end_dirs]
            for x in pool.imap_unordered(compose_images, interleave_directories(dirs, out_dir)):
                print(x)

        #do_one(r"E:\Simulations\CeramicBands\composed\adj_all_4k", end_dirs_round4)
        #do_one(r"E:\Simulations\CeramicBands\composed\adj_low_4k", end_dirs_adj_low)
        # do_one(r"E:\Simulations\CeramicBands\composed\crop_test_2", end_dirs_low2)
        # do_one(r"E:\Simulations\CeramicBands\composed\pure_med", pure_med)
        # do_one(r"E:\Simulations\CeramicBands\composed\threep_med", threep_med)
        # do_one(r"E:\Simulations\CeramicBands\composed\pure_fine", pure_fine)
        # do_one(r"E:\Simulations\CeramicBands\composed\threep_fine", threep_fine)
        # do_one(r"E:\Simulations\CeramicBands\composed\pure_fine_narrow", pure_fine_narrow)
        # do_one(r"E:\Simulations\CeramicBands\composed\pure_fine_low", pure_fine_low)
        # do_one(r"E:\Simulations\CeramicBands\composed\pure_fine_med", pure_fine_med)
        do_one(r"E:\Simulations\CeramicBands\composed\pure_fine_all_uncropped", pure_fine_all)



def _get_sim_infos():
    metas = glob.glob(r"E:\Simulations\CeramicBands\v7\pics\[9-A]*\Meta.txt")
    for fn in sorted(metas):
        dir_end = pathlib.Path(fn)

        try:
            sim_info = SimulationInfo.from_meta_txt(fn)

        except ValueError as e:
            sim_info = None

        yield sim_info, dir_end
        

def print_sim_infos():
    def sort_key(sim_info_dir_end):
        sim_info = sim_info_dir_end[0]
        return sim_info.source_file_name, sim_info.freedom_cases

    for sim_info, dir_end in sorted(_get_sim_infos(), key=sort_key):
        print(str(dir_end.parents[0]), sim_info)



if __name__ == "__main__":
    print_sim_infos()
    do_all_multi_process()

if __name__ == "__ASDASDASD__":
    test_distinctive_strings()
    test_compose_image()

    do_all_multi_process()

    sim_info = SimulationInfo.from_meta_txt(r"E:\Simulations\CeramicBands\v7\pics\70\Meta.txt")
    print(sim_info)

