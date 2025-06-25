import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import mixed_precision
import numpy as np
import albumentations as A
import os
import random
from glob import glob
import time
import datetime
import re

# ... (Ваш класс WarmupCosineDecay остается здесь без изменений) ...
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_target_lr, decay_steps, total_steps, warmup_steps, min_lr=0.0, warmup_start_factor=0.001, name=None):
        super().__init__()
        self.warmup_target_lr = warmup_target_lr; self.decay_steps = decay_steps
        self.total_steps = total_steps; self.warmup_steps = warmup_steps
        self.min_lr = min_lr; self.warmup_start_factor = warmup_start_factor; self.name = name
        if self.warmup_steps < 0: raise ValueError(f"Warmup steps ({self.warmup_steps}) must be non-negative.")
        if self.decay_steps < 0: raise ValueError(f"Decay steps ({self.decay_steps}) must be non-negative.")
        if self.warmup_steps + self.decay_steps == 0 and self.total_steps > 0 : pass
        elif self.warmup_steps > self.total_steps: raise ValueError(f"Warmup steps ({self.warmup_steps}) cannot exceed total training steps ({self.total_steps}).")
    def __call__(self, step):
        with tf.name_scope(self.name or "WarmupCosineDecay"):
            warmup_target_lr_t = tf.cast(self.warmup_target_lr, dtype=tf.float32); min_lr_t = tf.cast(self.min_lr, dtype=tf.float32)
            warmup_steps_t = tf.cast(self.warmup_steps, dtype=tf.float32); decay_steps_t = tf.cast(self.decay_steps, dtype=tf.float32)
            global_step_t = tf.cast(step, dtype=tf.float32); warmup_start_lr = warmup_target_lr_t * self.warmup_start_factor
            def warmup_phase_fn():
                slope = (warmup_target_lr_t - warmup_start_lr) / tf.maximum(warmup_steps_t, 1.0)
                return warmup_start_lr + slope * global_step_t
            def decay_phase_fn():
                current_decay_step = global_step_t - warmup_steps_t; effective_decay_steps = tf.maximum(decay_steps_t, 1.0)
                cosine_val = 0.5 * (1.0 + tf.cos(tf.constant(np.pi, dtype=tf.float32) * \
                                                 tf.minimum(current_decay_step, effective_decay_steps) / effective_decay_steps))
                lr_decayed = (warmup_target_lr_t - min_lr_t) * cosine_val + min_lr_t
                return tf.cond(tf.equal(decay_steps_t, 0.0), lambda: warmup_target_lr_t, lambda: lr_decayed)
            learning_rate = tf.cond(global_step_t < warmup_steps_t, warmup_phase_fn, decay_phase_fn)
            return tf.maximum(learning_rate, min_lr_t)
    def get_config(self):
        return {"warmup_target_lr": self.warmup_target_lr, "decay_steps": self.decay_steps, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps, "min_lr": self.min_lr, "warmup_start_factor": self.warmup_start_factor, "name": self.name}
# --- Конец класса WarmupCosineDecay ---

print(f"DEBUG: TensorFlow Version: {tf.__version__}")
print(f"DEBUG: Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

mixed_precision.set_global_policy('mixed_float16')

# --- ПАРАМЕТР ДЛЯ УКАЗАНИЯ ЧЕКПОИНТА ДЛЯ ЗАГРУЗКИ ---
# Укажите ПОЛНЫЙ ПУТЬ К ПРЕФИКСУ чекпоинта (например, './moco_checkpoints/model_name/ckpt-5')
# Оставьте пустым ('') или None, если не хотите загружать чекпоинт и хотите начать с нуля.
LOAD_SPECIFIC_CHECKPOINT_PREFIX = "/mnt/c/Users/urako/OneDrive/Документы/Code/Mr.-Mole/CNN/moco_checkpoints/original_mobilenet_encoder_bs64x1_q32768_t0.20/ckpt-80"
# LOAD_SPECIFIC_CHECKPOINT_PREFIX = "./moco_checkpoints/original_mobilenet_encoder_bs64x1_q32768_t0.10/ckpt-1" # Пример

# --- DEBUG PARAMS ---
DEBUG_MODE = False # Установите True для быстрой отладки
NUM_DEBUG_FILES = 16
SIMPLIFIED_MODEL = False
SIMPLIFIED_DATAPIPE = False
DISABLE_TF_FUNCTION = False

# ... (остальные параметры BATCH_SIZE, IMG_SIZE и т.д. остаются) ...
BATCH_SIZE = 64; ACCUMULATION_STEPS = 1; IMG_SIZE = (224,224); EMBED_DIM = 128
QUEUE_SIZE = 32768; TEMPERATURE = 0.2; MOMENTUM = 0.999
EPOCHS_SSL = 3 if DEBUG_MODE else 200
AUTOTUNE = tf.data.AUTOTUNE; RANDOM_SEED = 42; EPSILON_NORM = 1e-6
SSL_TARGET_LR = 1e-4; SSL_MIN_LR = 1e-6; SSL_WARMUP_EPOCHS = 5; SSL_WARMUP_START_FACTOR = 0.01

# ... (код загрузки данных all_filepaths и т.д. остается) ...
DATA_ROOT = "/mnt/a/Datasets"
all_filepaths_orig = glob(os.path.join(DATA_ROOT, "**", "*.jpg"), recursive=True)
random.seed(RANDOM_SEED); random.shuffle(all_filepaths_orig)
percentage_to_use = 0.15; num_files_to_use = int(len(all_filepaths_orig) * percentage_to_use)
all_filepaths_subset = all_filepaths_orig[:num_files_to_use]
if not all_filepaths_orig:
    print(f"WARNING: No JPG files found in {DATA_ROOT}.")
    if DEBUG_MODE: print(f"WARNING: Setting SIMPLIFIED_DATAPIPE=True."); SIMPLIFIED_DATAPIPE = True
    else: print(f"ERROR: No JPG files for normal run. Exiting."); exit()
if DEBUG_MODE:
    if all_filepaths_orig and not SIMPLIFIED_DATAPIPE: all_filepaths = all_filepaths_subset[:NUM_DEBUG_FILES]; print(f"DEBUG: Using {len(all_filepaths)} images.")
    elif not SIMPLIFIED_DATAPIPE: print("ERROR: DEBUG on, no real files, SIMPLIFIED_DATAPIPE False. Exit."); exit()
else:
    all_filepaths = all_filepaths_subset
    if not all_filepaths: print(f"ERROR: No JPG files for normal run. Exiting."); exit()
print(f"INFO: Using {len(all_filepaths)} files ({percentage_to_use*100:.1f}% of original).")

# ... (функции make_aug, preprocess, make_pair_numpy, tf_make_pair, tf_make_pair_simplified остаются) ...
def make_aug():
    return A.Compose([A.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0)),
                      A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                      A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                      A.GaussNoise(p=0.5)])
augmentor = make_aug()
def preprocess(path):
    img = tf.io.read_file(path); img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE); img = tf.cast(img, tf.float32) / 255.0
    return img
def make_pair_numpy(img_np):
    img_uint8 = (img_np * 255.0).astype(np.uint8)
    v1_np = augmentor(image=img_uint8)['image']; v2_np = augmentor(image=img_uint8)['image']
    return tf.cast(v1_np, tf.float32)/255.0, tf.cast(v2_np, tf.float32)/255.0
def tf_make_pair(path):
    img = preprocess(path)
    v1, v2 = tf.py_function(lambda x: make_pair_numpy(x.numpy()), [img], [tf.float32, tf.float32])
    v1.set_shape((*IMG_SIZE, 3)); v2.set_shape((*IMG_SIZE, 3)); return v1, v2
def tf_make_pair_simplified(dummy_input):
    return tf.random.uniform(shape=(*IMG_SIZE, 3)), tf.random.uniform(shape=(*IMG_SIZE, 3))

# ... (функции build_encoder_simplified, build_encoder_original остаются) ...
def build_encoder_simplified():
    inp = layers.Input(shape=(*IMG_SIZE,3)); x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D()(x); x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x); x = layers.Flatten()(x); x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(EMBED_DIM)(x); x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1, epsilon=EPSILON_NORM), dtype='float16')(x)
    print("DEBUG: Using SIMPLIFIED encoder."); return models.Model(inp, x, name="simplified_encoder")
def build_encoder_original():
    inp = layers.Input(shape=(*IMG_SIZE,3), name="input_mobilenet")
    base = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    x = base(inp) 
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(EMBED_DIM)(x)
    final_output = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1, epsilon=EPSILON_NORM), dtype='float16')(x)
    print("DEBUG: Using ORIGINAL MobileNetV3Small encoder (weights='imagenet')."); return models.Model(inp, final_output, name="original_mobilenet_encoder")

build_encoder = build_encoder_simplified if (DEBUG_MODE and SIMPLIFIED_MODEL) else build_encoder_original

print("DEBUG: Building encoder_q..."); encoder_q = build_encoder()
print("DEBUG: Building encoder_k..."); encoder_k = build_encoder()

print("DEBUG: Initializing queue...")
moco_queue_var = tf.Variable(initial_value=tf.math.l2_normalize(tf.random.normal([min(QUEUE_SIZE,128) if DEBUG_MODE else QUEUE_SIZE, EMBED_DIM],dtype=tf.float32),axis=1,epsilon=EPSILON_NORM),trainable=False,dtype=tf.float32,name="moco_queue")
moco_queue_ptr_var = tf.Variable(0, trainable=False, dtype=tf.int64, name="moco_queue_ptr")

# ... (функции moco_loss_impl, update_key_encoder_impl, dequeue_and_enqueue_impl и их @tf.function обертки остаются) ...
def moco_loss_impl(q,k_in):
    q_fp32=tf.cast(q,tf.float32);k_fp32=tf.cast(k_in,tf.float32)
    q_norm=tf.math.l2_normalize(q_fp32,axis=1,epsilon=EPSILON_NORM);k_norm=tf.math.l2_normalize(k_fp32,axis=1,epsilon=EPSILON_NORM)
    pos_logits=tf.reduce_sum(q_norm*k_norm,axis=1,keepdims=True);queue_fp32=tf.cast(moco_queue_var,tf.float32)
    neg_logits=tf.matmul(q_norm,queue_fp32,transpose_b=True);logits=tf.concat([pos_logits,neg_logits],axis=1);logits/=TEMPERATURE
    labels=tf.zeros(tf.shape(q_norm)[0],dtype=tf.int32);loss_per_sample=tf.keras.losses.sparse_categorical_crossentropy(labels,logits,from_logits=True)
    return tf.reduce_mean(loss_per_sample)
def update_key_encoder_impl():
    for qw,kw in zip(encoder_q.weights,encoder_k.weights):kw.assign(kw*MOMENTUM+qw*(1.-MOMENTUM))
def dequeue_and_enqueue_impl(keys):
    keys_fp32=tf.cast(keys,tf.float32);keys_normalized_fp32=tf.math.l2_normalize(keys_fp32,axis=1,epsilon=EPSILON_NORM)
    batch_size_val=tf.shape(keys_normalized_fp32)[0];ptr_val=moco_queue_ptr_var.read_value()
    batch_size64=tf.cast(batch_size_val,tf.int64);current_queue_size=tf.cast(tf.shape(moco_queue_var)[0],tf.int64)
    indices=tf.expand_dims(tf.range(ptr_val,ptr_val+batch_size64)%current_queue_size,1)
    updated_queue=tf.tensor_scatter_nd_update(moco_queue_var,indices,keys_normalized_fp32)
    moco_queue_var.assign(updated_queue);moco_queue_ptr_var.assign((ptr_val+batch_size64)%current_queue_size)

if DEBUG_MODE and DISABLE_TF_FUNCTION: print("DEBUG: @tf.function is DISABLED."); moco_loss=moco_loss_impl; update_key_encoder=update_key_encoder_impl; dequeue_and_enqueue=dequeue_and_enqueue_impl
else: print("DEBUG: @tf.function is ENABLED."); moco_loss=tf.function(moco_loss_impl); update_key_encoder=tf.function(update_key_encoder_impl); dequeue_and_enqueue=tf.function(dequeue_and_enqueue_impl)

# --- Настройка датасета ---
# ... (остается) ...
print("DEBUG: Setting up dataset pipeline...")
if DEBUG_MODE and SIMPLIFIED_DATAPIPE:
    print("DEBUG: Using SIMPLIFIED data pipeline."); dummy_elements_count = BATCH_SIZE*ACCUMULATION_STEPS*5 
    paths_ds = tf.data.Dataset.from_tensor_slices(tf.zeros(dummy_elements_count)); map_fn = tf_make_pair_simplified
else:
    if not all_filepaths: print("ERROR: all_filepaths empty. Exit."); exit()
    print(f"DEBUG: Using file-based data pipeline: {len(all_filepaths)} files."); paths_ds = tf.data.Dataset.from_tensor_slices(all_filepaths); map_fn = tf_make_pair
buffer_s = max(len(all_filepaths) if not (DEBUG_MODE and SIMPLIFIED_DATAPIPE) else dummy_elements_count, BATCH_SIZE*ACCUMULATION_STEPS*2)
paths_ds = paths_ds.shuffle(buffer_size=buffer_s, seed=RANDOM_SEED, reshuffle_each_iteration=True)
paths_ds = paths_ds.map(map_fn, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
print("DEBUG: Dataset pipeline configured.")

# --- Инициализация LR Scheduler и Оптимизатора ---
# ... (остается) ...
print("DEBUG: Initializing optimizer...")
if DEBUG_MODE and SIMPLIFIED_DATAPIPE: num_files_for_steps = dummy_elements_count
elif len(all_filepaths) > 0: num_files_for_steps = len(all_filepaths)
else: raise ValueError("Cannot calculate training steps.")
num_phys_batch_ep = num_files_for_steps//BATCH_SIZE or 1; num_eff_steps_ep = num_phys_batch_ep//ACCUMULATION_STEPS or 1
total_eff_steps = num_eff_steps_ep*EPOCHS_SSL or 1; warmup_eff_steps_calc = SSL_WARMUP_EPOCHS*num_eff_steps_ep
warmup_eff_steps = min(warmup_eff_steps_calc, max(0, total_eff_steps-1)); decay_eff_steps_calc = total_eff_steps-warmup_eff_steps
if decay_eff_steps_calc <= 0: decay_eff_steps_calc = 1; warmup_eff_steps = max(0, total_eff_steps-decay_eff_steps_calc)
print(f"INFO: Eff BS: {BATCH_SIZE*ACCUMULATION_STEPS}, Phys batch/ep: {num_phys_batch_ep}, Eff steps/ep: {num_eff_steps_ep}")
print(f"INFO: Total eff steps: {total_eff_steps}, Warmup: {warmup_eff_steps}, Decay: {decay_eff_steps_calc}")
lr_scheduler = WarmupCosineDecay(SSL_TARGET_LR, decay_eff_steps_calc, total_eff_steps, warmup_eff_steps, SSL_MIN_LR, SSL_WARMUP_START_FACTOR)
opt_base = tf.keras.optimizers.AdamW(learning_rate=lr_scheduler)
opt = mixed_precision.LossScaleOptimizer(opt_base)

# --- Настройка TensorBoard ---
# ... (остается) ...
summary_writer = None
if not DEBUG_MODE:
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S"); model_name_log = encoder_q.name
    log_dir_name = f"moco_{model_name_log}_bs{BATCH_SIZE}x{ACCUMULATION_STEPS}_q{QUEUE_SIZE}_t{TEMPERATURE:.2f}_ep{EPOCHS_SSL}"
    log_dir = os.path.join("logs", log_dir_name, time_str); summary_writer = tf.summary.create_file_writer(log_dir)
    print(f"INFO: TensorBoard logs: {log_dir}")

# --- Настройка и ЗАГРУЗКА/СОХРАНЕНИЕ Чекпоинтов ---
CKPT_DIR_BASE = "./moco_checkpoints"; model_name_ckpt = encoder_q.name
SAVE_CKPT_DIR = os.path.join(CKPT_DIR_BASE, f"{model_name_ckpt}_bs{BATCH_SIZE}x{ACCUMULATION_STEPS}_q{QUEUE_SIZE}_t{TEMPERATURE:.2f}")
SAVE_N_EPOCHS = 2 if not DEBUG_MODE else 1; MAX_CKPTS_TO_KEEP = 3
if not DEBUG_MODE or (DEBUG_MODE and SAVE_N_EPOCHS > 0):
    os.makedirs(SAVE_CKPT_DIR, exist_ok=True); print(f"INFO: Checkpoints will be saved to: {SAVE_CKPT_DIR}")

opt_to_ckpt = opt.inner_optimizer if isinstance(opt, mixed_precision.LossScaleOptimizer) else opt
ckpt_obj = tf.train.Checkpoint(step=tf.Variable(0,dtype=tf.int64), epoch=tf.Variable(0,dtype=tf.int64),
                             optimizer=opt_to_ckpt, encoder_q=encoder_q, encoder_k=encoder_k,
                             moco_queue=moco_queue_var, moco_queue_ptr=moco_queue_ptr_var)
# CheckpointManager для СОХРАНЕНИЯ
ckpt_manager = tf.train.CheckpointManager(ckpt_obj, SAVE_CKPT_DIR, max_to_keep=MAX_CKPTS_TO_KEEP)

initial_epoch = 0
if LOAD_SPECIFIC_CHECKPOINT_PREFIX and os.path.exists(LOAD_SPECIFIC_CHECKPOINT_PREFIX + ".index"):
    print(f"INFO: Attempting to restore from specified checkpoint: {LOAD_SPECIFIC_CHECKPOINT_PREFIX}")
    try:
        # Загружаем напрямую из указанного файла (префикса)
        status = ckpt_obj.restore(LOAD_SPECIFIC_CHECKPOINT_PREFIX).expect_partial()
        # status.assert_existing_objects_matched() # Для строгой проверки, если нужно
        initial_epoch = ckpt_obj.epoch.numpy()
        print(f"INFO: Successfully restored from {LOAD_SPECIFIC_CHECKPOINT_PREFIX}. Resuming from epoch {initial_epoch}, optimizer step {ckpt_obj.step.numpy()}")
    except Exception as e:
        print(f"ERROR: Failed to restore checkpoint from {LOAD_SPECIFIC_CHECKPOINT_PREFIX}. Error: {e}")
        print("INFO: Starting from scratch as specified checkpoint failed to load.")
        encoder_k.set_weights(encoder_q.get_weights()) # Инициализация по умолчанию, если загрузка не удалась
else:
    if LOAD_SPECIFIC_CHECKPOINT_PREFIX: # Если путь был указан, но файл не найден
        print(f"WARNING: Specified checkpoint prefix not found: {LOAD_SPECIFIC_CHECKPOINT_PREFIX}. Starting from scratch.")
    else: # Если путь не был указан
        print("INFO: No specific checkpoint path provided. Starting from scratch.")
    encoder_k.set_weights(encoder_q.get_weights()) # Инициализация по умолчанию

print("DEBUG: Starting training loop with Gradient Accumulation...")
accum_grads = [tf.zeros_like(w, dtype=tf.float32) for w in encoder_q.trainable_weights]

for epoch_loop_idx in range(initial_epoch, EPOCHS_SSL):
    # ... (остальной цикл обучения остается таким же, как в вашем последнем коде) ...
    # Важно: используйте ckpt_obj.step и ckpt_obj.epoch для отслеживания и сохранения
    current_epoch_num_display = epoch_loop_idx + 1
    ckpt_obj.epoch.assign(current_epoch_num_display)

    print(f"DEBUG: --- Epoch {current_epoch_num_display}/{EPOCHS_SSL} (Opt Step: {ckpt_obj.step.numpy()}) ---")
    ep_start_time = time.time(); batch_times = []; ep_loss_sum_avg = 0.0; eff_steps_this_ep = 0

    for batch_idx, (v1, v2) in enumerate(paths_ds):
        # batch_start_time = time.time()
        with tf.GradientTape() as tape:
            q_out = encoder_q(v1, training=True); k_out = encoder_k(v2, training=False)
            k_stop = tf.stop_gradient(k_out); loss_curr = moco_loss(q_out, k_stop)
        grads_micro = tape.gradient(loss_curr, encoder_q.trainable_weights)
        for i in range(len(accum_grads)):
            if grads_micro[i] is not None: accum_grads[i] += grads_micro[i]
        
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            eff_grads = [g/ACCUMULATION_STEPS for g in accum_grads]
            opt.apply_gradients(zip(eff_grads, encoder_q.trainable_weights)); ckpt_obj.step.assign_add(1)
            accum_grads = [tf.zeros_like(w,dtype=tf.float32) for w in encoder_q.trainable_weights]
            update_key_encoder(); dequeue_and_enqueue(k_stop)
            
            loss_val_log = loss_curr.numpy(); ep_loss_sum_avg += loss_val_log*ACCUMULATION_STEPS; eff_steps_this_ep += 1
            if summary_writer:
                with summary_writer.as_default():
                    tf.summary.scalar('moco_loss_eff_step', loss_val_log, step=ckpt_obj.step.numpy())
                    lr_val = lr_scheduler(ckpt_obj.step); tf.summary.scalar('learning_rate', lr_val, step=ckpt_obj.step.numpy())
            if ckpt_obj.step.numpy()%20==0 or (DEBUG_MODE and ckpt_obj.step.numpy()<=5):
                lr_print = lr_scheduler(ckpt_obj.step).numpy()
                print(f"INFO: Ep {current_epoch_num_display}, PhysBatch {batch_idx+1}, EffStep {ckpt_obj.step.numpy()}. Loss(micro): {loss_val_log:.4f}, LR: {lr_print:.2e}")
        
        # batch_end_time = time.time(); batch_times.append(batch_end_time - batch_start_time)
        loss_check = loss_curr.numpy()
        if np.isnan(loss_check) or np.isinf(loss_check):
            print(f"ERROR: Loss {loss_check} at phys batch {batch_idx+1}. Stop.");
            if ckpt_manager and not DEBUG_MODE: ckpt_manager.save(); print(f"INFO: Emergency ckpt saved.")
            break 
        if DEBUG_MODE:
            max_debug_eff = (NUM_DEBUG_FILES//(BATCH_SIZE*ACCUMULATION_STEPS))+1 if not SIMPLIFIED_DATAPIPE else 2
            if SIMPLIFIED_DATAPIPE and max_debug_eff<1: max_debug_eff=1
            if ckpt_obj.step.numpy() >= max_debug_eff:
                print(f"DEBUG: Max debug eff steps ({max_debug_eff}) reached. Break.")
                if (batch_idx+1)%ACCUMULATION_STEPS != 0 and any(tf.reduce_sum(tf.abs(g)).numpy()>0 for g in accum_grads):
                    num_rem = (batch_idx+1)%ACCUMULATION_STEPS or ACCUMULATION_STEPS
                    eff_grads_dbg = [g/num_rem for g in accum_grads]
                    opt.apply_gradients(zip(eff_grads_dbg, encoder_q.trainable_weights)); ckpt_obj.step.assign_add(1)
                    update_key_encoder(); dequeue_and_enqueue(k_stop)
                break 
    
    num_rem_ep_end = (batch_idx+1)%ACCUMULATION_STEPS # batch_idx здесь - последний обработанный
    # Если цикл прервался из-за NaN, то batch_idx может быть не последним в эпохе
    # Проверяем, есть ли накопленные градиенты и не был ли последний шаг уже эффективным
    if not (np.isnan(loss_check) or np.isinf(loss_check)) and \
       num_rem_ep_end != 0 and \
       any(tf.reduce_sum(tf.abs(g)).numpy() > 0 for g in accum_grads):
        print(f"DEBUG: Applying {num_rem_ep_end} remaining grads at epoch end...")
        eff_grads_ep = [g/num_rem_ep_end for g in accum_grads]
        opt.apply_gradients(zip(eff_grads_ep, encoder_q.trainable_weights)); ckpt_obj.step.assign_add(1)
        update_key_encoder()
        if 'k_stop' in locals(): dequeue_and_enqueue(k_stop)
        if summary_writer:
            with summary_writer.as_default():
                loss_val_log = loss_curr.numpy() if 'loss_curr' in locals() else 0.0
                ep_loss_sum_avg += loss_val_log*num_rem_ep_end; eff_steps_this_ep +=1
                tf.summary.scalar('moco_loss_eff_step', loss_val_log, step=ckpt_obj.step.numpy())
                lr_val = lr_scheduler(ckpt_obj.step); tf.summary.scalar('learning_rate', lr_val, step=ckpt_obj.step.numpy())

    ep_end_time = time.time(); avg_micro_batch_t = np.mean(batch_times) if batch_times else 0
    avg_ep_loss = ep_loss_sum_avg / (eff_steps_this_ep*ACCUMULATION_STEPS) if eff_steps_this_ep > 0 else 0.0
    print(f"Epoch {current_epoch_num_display}/{EPOCHS_SSL}, AvgLoss={avg_ep_loss:.4f}, EpTime: {ep_end_time-ep_start_time:.2f}s, AvgMicroBatchT: {avg_micro_batch_t:.2f}s, OptSteps: {ckpt_obj.step.numpy()}")
    
    if not DEBUG_MODE and SAVE_N_EPOCHS > 0:
        if (current_epoch_num_display % SAVE_N_EPOCHS == 0) or (current_epoch_num_display == EPOCHS_SSL):
            save_path = ckpt_manager.save(); print(f"INFO: Saved ckpt for ep {current_epoch_num_display} (step {ckpt_obj.step.numpy()}) to {save_path}")
    
    if 'loss_check' in locals() and (np.isnan(loss_check) or np.isinf(loss_check)): print("ERROR: NaN/Inf loss. Stop."); break
    if DEBUG_MODE:
        print("DEBUG: DEBUG Epoch done. Stop.")
        if ckpt_manager and SAVE_N_EPOCHS > 0 : save_path = ckpt_manager.save(); print(f"INFO: Saved DEBUG ckpt for ep {current_epoch_num_display} (step {ckpt_obj.step.numpy()}) to {save_path}")
        break

# Определяем, было ли обучение завершено "нормально" для целей сохранения финальной модели
save_final_encoder_flag = False
last_completed_epoch_num = ckpt_obj.epoch.numpy()

# Сначала проверяем, не было ли прерывания из-за ошибки NaN/Inf
if 'loss_check' in locals() and (np.isnan(loss_check) or np.isinf(loss_check)):
    print("INFO: Training was interrupted by NaN/Inf loss. Final encoder_q will not be saved as a separate .keras file.")
else:
    # Если ошибок не было, проверяем, завершились ли все запланированные эпохи
    # или это был DEBUG_MODE, который завершился (даже если раньше по шагам, но без ошибок).
    if not DEBUG_MODE:
        if last_completed_epoch_num >= EPOCHS_SSL: # Условие >= на случай, если обучение пошло дальше из-за initial_epoch
            save_final_encoder_flag = True
            print(f"INFO: Main SSL training completed all {EPOCHS_SSL} epochs (last completed: {last_completed_epoch_num}).")
        else:
            print(f"INFO: Main SSL training did not complete all {EPOCHS_SSL} epochs (last completed: {last_completed_epoch_num}). "
                  f"Final encoder_q will not be saved separately as a '.keras' file.")
    else:
        # В DEBUG_MODE, если не было NaN/Inf, мы считаем любой выход из цикла "успешным завершением дебага".
        save_final_encoder_flag = True
        print(f"INFO: DEBUG SSL training run completed (last completed epoch: {last_completed_epoch_num} "
              f"of {EPOCHS_SSL} planned for debug).")

if save_final_encoder_flag and SAVE_CKPT_DIR:
    print(f"INFO: Saving the final query encoder (encoder_q) to a .keras file...")
    try:
        # Формируем имя файла
        model_save_name = f"final_encoder_q_epoch{last_completed_epoch_num}"
        if DEBUG_MODE:
            model_save_name += "_debug"
        
        final_encoder_path = os.path.join(SAVE_CKPT_DIR, f"{model_save_name}.keras")
        
        # Сохраняем только encoder_q, так как это обычно то, что используется для downstream задач
        encoder_q.save(final_encoder_path)
        print(f"INFO: Query encoder (encoder_q) saved in Keras format to: {final_encoder_path}")

    except Exception as e:
        print(f"ERROR: Could not save the final encoder_q model as a .keras file. Error: {e}")

if not DEBUG_MODE and ckpt_manager.latest_checkpoint: print(f"INFO: Training done. Last ckpt: {ckpt_manager.latest_checkpoint}")
elif DEBUG_MODE and ckpt_manager.latest_checkpoint: print(f"INFO: DEBUG training done. Last ckpt: {ckpt_manager.latest_checkpoint}")
else: print("DEBUG: No ckpts managed/saved or training interrupted early.")

if summary_writer: summary_writer.close()
print("DEBUG: Script finished.")