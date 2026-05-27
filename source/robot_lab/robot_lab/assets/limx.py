import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

usd_dir_path = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots/limx/HU_D03_description/usd/")
robot_usd = "HU_D03_03.usd"

HU_D03_03 = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path = usd_dir_path + robot_usd,
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity = False,
            # retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity = 2.0,
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            fix_root_link = False,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.91),
        # joint_pos={
        #     'left_hip_pitch_joint': -0.3,
        #     'left_hip_roll_joint': -0.01,
        #     'left_hip_yaw_joint': -0.1,
        #     'left_knee_joint': 0.6,
        #     'left_ankle_pitch_joint': -0.35,
        #     'left_ankle_roll_joint': 0.0,

        #     'right_hip_pitch_joint': -0.3,
        #     'right_hip_roll_joint': 0.01,
        #     'right_hip_yaw_joint': 0.1,
        #     'right_knee_joint': 0.6,
        #     'right_ankle_pitch_joint': -0.35,
        #     'right_ankle_roll_joint': 0.0,

        #     'waist_yaw_joint': 0.0,
        #     'waist_roll_joint': 0.0,
        #     'waist_pitch_joint': 0.0,

        #     'head_yaw_joint': 0.0,
        #     'head_pitch_joint': 0.0,

        #     'left_shoulder_pitch_joint': 0.1,
        #     'left_shoulder_roll_joint': 0.1,
        #     'left_shoulder_yaw_joint': -0.2,
        #     'left_elbow_joint': -0.2,
        #     # 'left_hand_roll_joint': 0.0,
        #     # 'left_hand_pitch_joint': 0.0,
        #     'left_wrist_yaw_joint': 0.0,
        #     'left_wrist_pitch_joint': 0.0,
        #     'left_hand_yaw_joint': 0.0,

        #     'right_shoulder_pitch_joint': 0.1,
        #     'right_shoulder_roll_joint': -0.1,
        #     'right_shoulder_yaw_joint': 0.2,
        #     'right_elbow_joint': -0.2,
        #     # 'right_hand_roll_joint': 0.0,
        #     # 'right_hand_pitch_joint': 0.0,
        #     'right_wrist_yaw_joint': 0.0,
        #     'right_wrist_pitch_joint': 0.0,
        #     'right_hand_yaw_joint': 0.0,
        # },
        # joint_pos={
        #     'left_hip_pitch_joint': 0.,
        #     'left_hip_roll_joint': 0.,
        #     'left_hip_yaw_joint': -0.,
        #     'left_knee_joint': 0.,
        #     'left_ankle_pitch_joint': 0.,
        #     'left_ankle_roll_joint': 0.,

        #     'right_hip_pitch_joint': 0.,
        #     'right_hip_roll_joint': 0.,
        #     'right_hip_yaw_joint': 0.,
        #     'right_knee_joint': 0.,
        #     'right_ankle_pitch_joint': 0.,
        #     'right_ankle_roll_joint': 0.,

        #     'waist_yaw_joint': 0.0,
        #     'waist_roll_joint': 0.0,
        #     'waist_pitch_joint': 0.0,

        #     'head_yaw_joint': 0.0,
        #     'head_pitch_joint': 0.0,

        #     'left_shoulder_pitch_joint': 0.1,
        #     'left_shoulder_roll_joint': 0.1,
        #     'left_shoulder_yaw_joint': -0.2,
        #     'left_elbow_joint': -0.2,
        #     # 'left_hand_roll_joint': 0.0,
        #     # 'left_hand_pitch_joint': 0.0,
        #     'left_wrist_yaw_joint': 0.0,
        #     'left_wrist_pitch_joint': 0.0,
        #     'left_hand_yaw_joint': 0.0,

        #     'right_shoulder_pitch_joint': 0.1,
        #     'right_shoulder_roll_joint': -0.1,
        #     'right_shoulder_yaw_joint': 0.2,
        #     'right_elbow_joint': -0.2,
        #     # 'right_hand_roll_joint': 0.0,
        #     # 'right_hand_pitch_joint': 0.0,
        #     'right_wrist_yaw_joint': 0.0,
        #     'right_wrist_pitch_joint': 0.0,
        #     'right_hand_yaw_joint': 0.0,
        # },
        joint_pos={
            'left_hip_pitch_joint': -0.15,
            'left_hip_roll_joint': -0.0,
            'left_hip_yaw_joint': -0.05,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.16,
            'left_ankle_roll_joint': 0.0,

            'right_hip_pitch_joint': -0.15,
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.05,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.16,
            'right_ankle_roll_joint': 0.0,

            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,

            'head_yaw_joint': 0.0,
            'head_pitch_joint': 0.0,

            'left_shoulder_pitch_joint': 0.1,
            'left_shoulder_roll_joint': 0.1,
            'left_shoulder_yaw_joint': -0.2,
            'left_elbow_joint': -0.2,
            # 'left_hand_roll_joint': 0.0,
            # 'left_hand_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_hand_yaw_joint': 0.0,

            'right_shoulder_pitch_joint': 0.1,
            'right_shoulder_roll_joint': -0.1,
            'right_shoulder_yaw_joint': 0.2,
            'right_elbow_joint': -0.2,
            # 'right_hand_roll_joint': 0.0,
            # 'right_hand_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_hand_yaw_joint': 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "robot": ImplicitActuatorCfg(
            joint_names_expr=[
                '.*_hip_pitch_joint',
                '.*_hip_roll_joint',
                '.*_hip_yaw_joint',
                '.*_knee_joint',

                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",

                'waist_yaw_joint',
                'waist_roll_joint',
                'waist_pitch_joint',

                'head_pitch_joint',
                'head_yaw_joint',

                '.*_shoulder_pitch_joint',
                '.*_shoulder_roll_joint',
                '.*_shoulder_yaw_joint',
                '.*_elbow_joint',

                # '.*_hand_roll_joint',
                # '.*_hand_pitch_joint',
                '.*_wrist_yaw_joint',
                '.*_wrist_pitch_joint',
                '.*_hand_yaw_joint',
            ],
            stiffness={
                '.*_hip_pitch_joint': 144.,
                '.*_hip_roll_joint': 200.,
                '.*_hip_yaw_joint': 160.,
                '.*_knee_joint': 230.,

                ".*_ankle_pitch_joint": 20,
                ".*_ankle_roll_joint": 20,

                'waist_yaw_joint': 100.,
                'waist_roll_joint': 140.,
                'waist_pitch_joint': 140.,

                'head_pitch_joint': 5.0,
                'head_yaw_joint': 5.0,

                '.*_shoulder_pitch_joint': 64,
                '.*_shoulder_roll_joint': 80,
                '.*_shoulder_yaw_joint': 80,
                '.*_elbow_joint': 100,

                '.*_wrist_yaw_joint': 40,
                '.*_wrist_pitch_joint': 40,
                '.*_hand_yaw_joint': 40,
            },
            damping={
                '.*_hip_pitch_joint': 4.,
                '.*_hip_roll_joint': 3.5,
                '.*_hip_yaw_joint': 4.,
                '.*_knee_joint': 3.5,

                ".*_ankle_pitch_joint": 2,
                ".*_ankle_roll_joint": 2,

                'waist_yaw_joint': 7.5,
                'waist_roll_joint': 5.,
                'waist_pitch_joint': 5.,

                'head_pitch_joint': 0.5,
                'head_yaw_joint': 0.5,

                '.*_shoulder_pitch_joint': 4,
                '.*_shoulder_roll_joint': 4,
                '.*_shoulder_yaw_joint': 4,
                '.*_elbow_joint': 4,

                '.*_wrist_yaw_joint': 3,
                '.*_wrist_pitch_joint': 3,
                '.*_hand_yaw_joint': 3,
            },
             effort_limit={
                '.*_hip_pitch_joint': 120.,
                '.*_hip_roll_joint': 120.,
                '.*_hip_yaw_joint': 120.,
                '.*_knee_joint': 120.,

                ".*_ankle_pitch_joint": 80,
                ".*_ankle_roll_joint": 40,

                'waist_yaw_joint': 23,
                'waist_roll_joint': 40,
                'waist_pitch_joint': 40,

                'head_pitch_joint': 10.0,
                'head_yaw_joint': 10.0,

                '.*_shoulder_pitch_joint': 15.5,
                '.*_shoulder_roll_joint': 15.5,
                '.*_shoulder_yaw_joint': 15.5,
                '.*_elbow_joint': 15.5,

                '.*_wrist_yaw_joint': 10,
                '.*_wrist_pitch_joint': 10,
                '.*_hand_yaw_joint': 10,
            },
            velocity_limit={
                '.*_hip_pitch_joint': 12,
                '.*_hip_roll_joint': 12,
                '.*_hip_yaw_joint': 12,
                '.*_knee_joint': 12,

                ".*_ankle_pitch_joint": 16,
                ".*_ankle_roll_joint": 16,

                'waist_yaw_joint': 16.0,
                'waist_roll_joint': 16.0,
                'waist_pitch_joint': 16.0,

                'head_pitch_joint': 15,
                'head_yaw_joint': 15,

                '.*_shoulder_pitch_joint': 16,
                '.*_shoulder_roll_joint': 16,
                '.*_shoulder_yaw_joint': 16,
                '.*_elbow_joint': 16,

                '.*_wrist_yaw_joint': 15,
                '.*_wrist_pitch_joint': 15,
                '.*_hand_yaw_joint': 15,
            },
            armature={
                '.*_hip_pitch_joint': 0.15257125,
                '.*_hip_roll_joint': 0.15257125,
                '.*_hip_yaw_joint': 0.15257125,
                '.*_knee_joint': 0.15257125,

                ".*_ankle_pitch_joint": 0.094889232,
                ".*_ankle_roll_joint": 0.094889232,

                'waist_yaw_joint': 0.094889232,
                'waist_roll_joint': 0.094889232,
                'waist_pitch_joint': 0.094889232,

                'head_pitch_joint': 0.0106,
                'head_yaw_joint': 0.0106,

                '.*_shoulder_pitch_joint': 0.045760625,
                '.*_shoulder_roll_joint': 0.045760625,
                '.*_shoulder_yaw_joint': 0.045760625,
                '.*_elbow_joint': 0.045760625,

                '.*_wrist_yaw_joint': 0.0106,
                '.*_wrist_pitch_joint': 0.0106,
                '.*_hand_yaw_joint': 0.0106,
            },
            friction={
                '.*_hip_pitch_joint': 0.0,
                '.*_hip_roll_joint': 0.0,
                '.*_hip_yaw_joint': 0.0,
                '.*_knee_joint': 0.0,

                ".*_ankle_pitch_joint": 0.0,
                ".*_ankle_roll_joint": 0.0,

                'waist_yaw_joint': 0.0,
                'waist_roll_joint': 0.0,
                'waist_pitch_joint': 0.0,

                'head_pitch_joint': 0.0,
                'head_yaw_joint': 0.0,

                '.*_shoulder_pitch_joint': 0.0,
                '.*_shoulder_roll_joint': 0.0,
                '.*_shoulder_yaw_joint': 0.0,
                '.*_elbow_joint': 0.0,

                '.*_wrist_yaw_joint': 0.0,
                '.*_wrist_pitch_joint': 0.0,
                '.*_hand_yaw_joint': 0.0,
            },
        ),
    },
)
