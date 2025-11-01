from myosuite.utils import gym

def make_video_env(env_id, seed, video_dir, episode_trigger):
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=episode_trigger)
    env.reset(seed=seed)
    return env
