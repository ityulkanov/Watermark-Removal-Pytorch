from torch import optim
from torch import cuda
from tqdm.auto import tqdm
from model.generator import SkipEncoderDecoder, input_noise
from helper import *


def remove_watermark(image_path, mask_path, max_dim, reg_noise, input_depth, lr, show_step, training_steps,
                     tqdm_length=100):
    import torch
    DTYPE = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if not torch.cuda.is_available():
        print('\nSetting device to "cpu", since torch is not built with "cuda" support...')
        print('It is recommended to use GPU if possible...')

    image_np, mask_np = preprocess_images(image_path, mask_path, max_dim)

    print('Building the model...')
    get_less_used_gpu()
    generator = SkipEncoderDecoder(
        input_depth,
        num_channels_down=[128] * 5,
        num_channels_up=[128] * 5,
        num_channels_skip=[128] * 5
    ).type(DTYPE)

    objective = torch.nn.MSELoss().type(DTYPE)
    optimizer = optim.Adam(generator.parameters(), lr)

    image_var = np_to_torch_array(image_np).type(DTYPE)
    mask_var = np_to_torch_array(mask_np).type(DTYPE)

    generator_input = input_noise(input_depth, image_np.shape[1:]).type(DTYPE)

    generator_input_saved = generator_input.detach().clone()
    noise = generator_input.detach().clone()

    print('\nStarting training...\n')

    progress_bar = tqdm(range(training_steps), desc='Completed', ncols=tqdm_length)

    for step in progress_bar:
        optimizer.zero_grad()
        generator_input = generator_input_saved

        if reg_noise > 0:
            generator_input = generator_input_saved + (noise.normal_() * reg_noise)

        output = generator(generator_input)

        loss = objective(output * mask_var, image_var * mask_var)
        loss.backward()

        if step % show_step == 0:
            output_image = torch_to_np_array(output)
            visualize_sample(image_np, output_image, nrow=2, size_factor=10)

        progress_bar.set_postfix(Loss=loss.item())

        optimizer.step()

    output_image = torch_to_np_array(output)
    visualize_sample(output_image, nrow=1, size_factor=10)

    pil_image = Image.fromarray((output_image.transpose(1, 2, 0) * 255.0).astype('uint8'))

    output_path = image_path.split('/')[-1].split('.')[-2] + '-output.jpg'
    print(f'\nSaving final output image to: "{output_path}"\n')

    pil_image.save(output_path)


def get_less_used_gpu(gpus=None, debug=True):
    """Inspect cached/reserved and allocated memory on specified gpus and return the id of the less used device"""
    if gpus is None:
        warn = 'Falling back to default: all gpus'
        gpus = range(cuda.device_count())
    elif isinstance(gpus, str):
        gpus = [int(el) for el in gpus.split(',')]

    # check gpus arg VS available gpus
    sys_gpus = list(range(cuda.device_count()))
    if len(gpus) > len(sys_gpus):
        gpus = sys_gpus
        warn = f'WARNING: Specified {len(gpus)} gpus, but only {cuda.device_count()} available. Falling back to default: all gpus.\nIDs:\t{list(gpus)}'
    elif set(gpus).difference(sys_gpus):
        # take correctly specified and add as much bad specifications as unused system gpus
        available_gpus = set(gpus).intersection(sys_gpus)
        unavailable_gpus = set(gpus).difference(sys_gpus)
        unused_gpus = set(sys_gpus).difference(gpus)
        gpus = list(available_gpus) + list(unused_gpus)[:len(unavailable_gpus)]
        warn = f'GPU ids {unavailable_gpus} not available. Falling back to {len(gpus)} device(s).\nIDs:\t{list(gpus)}'

    cur_allocated_mem = {}
    cur_cached_mem = {}
    max_allocated_mem = {}
    max_cached_mem = {}
    for i in gpus:
        cur_allocated_mem[i] = cuda.memory_allocated(i)
        cur_cached_mem[i] = cuda.memory_reserved(i)
        max_allocated_mem[i] = cuda.max_memory_allocated(i)
        max_cached_mem[i] = cuda.max_memory_reserved(i)
    min_allocated = min(cur_allocated_mem, key=cur_allocated_mem.get)
    if debug:
        print(warn)
        print('Current allocated memory:', {f'cuda:{k}': v for k, v in cur_allocated_mem.items()})
        print('Current reserved memory:', {f'cuda:{k}': v for k, v in cur_cached_mem.items()})
        print('Maximum allocated memory:', {f'cuda:{k}': v for k, v in max_allocated_mem.items()})
        print('Maximum reserved memory:', {f'cuda:{k}': v for k, v in max_cached_mem.items()})
        print('Suggested GPU:', min_allocated)
    return min_allocated


def free_memory(to_delete: list, debug=False):
    import gc
    import inspect
    calling_namespace = inspect.currentframe().f_back
    if debug:
        print('Before:')
        get_less_used_gpu(debug=True)

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        cuda.empty_cache()
    if debug:
        print('After:')
        get_less_used_gpu(debug=True)
