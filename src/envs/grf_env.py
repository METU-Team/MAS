"""
envs/grf_env.py
===============
Google Research Football (GRF) ortam sarmalayıcısı.
Multi-agent gözlemlerini PyTorch tensorlerine dönüştürür.

Senaryo : academy_3_vs_1_with_keeper  (3 hücumcu vs 1 defans + kaleci)
Durum   : simple115v2  (115 boyutlu vektör, her ajan için ayrı gözlem)
"""

import os
import numpy as np
import torch
import gfootball.env as football_env
from typing import Optional


class GRFWrapper:
    """
    GRF ortamını sarmalayan ve multi-agent gözlemlerini PyTorch
    tensorlerine çeviren wrapper sınıfı.

    Ortam ``number_of_left_players_agent_controls > 1`` ile oluşturulduğunda
    gözlem dizisi ``(num_agents, obs_dim)`` şeklinde döner. Bu sınıf:
      - Her adımda bireysel gözlemleri ``torch.FloatTensor``'a çevirir.
      - Tüm gözlemlerin birleşimi olan global *state* vektörünü üretir.
      - Ödülleri ve bitiş sinyallerini uygun formata getirir.
    """

    def __init__(self, env, num_agents: int = 3, device: str = "cpu"):
        self.env = env
        self.num_agents = num_agents
        self.device = torch.device(device)

        # simple115v2 → 115 boyut; ortamdan dinamik okuma
        sample_obs = env.observation_space.shape
        # Multi-agent için shape (num_agents, obs_dim) veya sadece (obs_dim,)
        if len(sample_obs) == 2:
            self.obs_dim = sample_obs[1]
        else:
            self.obs_dim = sample_obs[0]

        self.n_actions = env.action_space.nvec[0] if hasattr(
            env.action_space, "nvec"
        ) else env.action_space.n

        # Global state boyutu: tüm ajan gözlemlerinin birleşimi
        self.state_dim = self.obs_dim * self.num_agents

    # ----- yardımcı dönüşümler -----
    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _parse_obs(self, obs: np.ndarray):
        """
        Ham gözlemi ayrıştırır.

        Dönüş
        ------
        agent_obs : Tensor  (num_agents, obs_dim)
        state     : Tensor  (state_dim,)   – tüm gözlemlerin birleşimi
        """
        obs = np.asarray(obs, dtype=np.float32)

        if obs.ndim == 1:
            # Tek ajanlı durum: boyutu genişlet
            obs = obs[np.newaxis, :]

        agent_obs = self._to_tensor(obs)                  # (n_agents, obs_dim)
        state = agent_obs.reshape(-1)                     # (state_dim,)
        return agent_obs, state

    # ----- ortam arayüzü -----
    def reset(self):
        """
        Ortamı sıfırlar.

        Dönüş
        ------
        agent_obs : Tensor  (num_agents, obs_dim)
        state     : Tensor  (state_dim,)
        """
        obs = self.env.reset()
        return self._parse_obs(obs)

    def step(self, actions):
        """
        Verilen eylemleri ortama uygular.

        Parametreler
        ----------
        actions : list[int] | np.ndarray  – her ajan için ayrık eylem indeksi.

        Dönüş
        ------
        agent_obs  : Tensor  (num_agents, obs_dim)
        state      : Tensor  (state_dim,)
        rewards    : Tensor  (num_agents,)
        done       : bool
        info       : dict
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        actions = np.asarray(actions, dtype=np.int32)

        obs, reward, done, info = self.env.step(actions)

        agent_obs, state = self._parse_obs(obs)

        # Ödül skalar gelirse her ajana aynı şekilde yayılır
        if np.isscalar(reward):
            rewards = self._to_tensor(
                np.full(self.num_agents, reward, dtype=np.float32)
            )
        else:
            rewards = self._to_tensor(np.asarray(reward, dtype=np.float32))

        return agent_obs, state, rewards, done, info

    def close(self):
        self.env.close()

    def get_env_info(self) -> dict:
        """Ortam meta bilgilerini döndürür."""
        return {
            "num_agents": self.num_agents,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
        }


# ---------------------------------------------------------------------------
# Fabrika fonksiyonu
# ---------------------------------------------------------------------------
def make_grf_env(
    env_name: str = "academy_3_vs_1_with_keeper",
    num_agents: int = 3,
    representation: str = "simple115v2",
    rewards: str = "scoring,checkpoints",
    render: bool = False,
    disable_3d: bool = False,
    dump_dir: Optional[str] = None,
    write_video: bool = False,
    device: str = "cpu",
) -> GRFWrapper:
    """
    GRF ortamını oluşturur ve ``GRFWrapper`` ile sarmalayıp döndürür.

    Parametreler
    ----------
    env_name        : Akademi senaryo adı.
    num_agents      : Sol takımda kontrol edilecek oyuncu sayısı.
    representation  : Gözlem temsili (``'simple115v2'`` vb.).
    rewards         : Ödül şeması.
    render          : Canlı render penceresi (WSL'de False bırakın).
    disable_3d      : True ise GRF 3D rendering devre disi kalir.
    dump_dir        : Video / trace dosyaları için klasör.
    write_video     : Video kaydı yapılsın mı.
    device          : PyTorch cihazı (``'cpu'`` veya ``'cuda'``).

    Dönüş
    ------
    GRFWrapper
    """
    if dump_dir is None:
        dump_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "videos",
        )
    os.makedirs(dump_dir, exist_ok=True)

    # Headless calisma: render kapaliyken pencere olusumunu engelle.
    # Bazi GRF/SDL surumlerinde render=False olsa bile pencere acilabilir.
    if not render:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    env = football_env.create_environment(
        env_name=env_name,
        stacked=False,
        representation=representation,
        rewards=rewards,
        number_of_left_players_agent_controls=num_agents,
        render=render,
        other_config_options={"disable_3d": disable_3d},
        write_video=write_video,
        logdir=dump_dir,
        write_full_episode_dumps=False,
        write_goal_dumps=False,
    )

    wrapper = GRFWrapper(env, num_agents=num_agents, device=device)

    print(f"[INFO] GRF ortamı oluşturuldu: {env_name}")
    print(f"       Ajan sayısı  : {num_agents}")
    print(f"       Gözlem boyutu: {wrapper.obs_dim}")
    print(f"       State boyutu : {wrapper.state_dim}")
    print(f"       Eylem sayısı : {wrapper.n_actions}")

    return wrapper
