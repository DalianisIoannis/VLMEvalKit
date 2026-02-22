from ..smp import *

FAIL_MSG = 'Failed to obtain answer via API.'

class LLavaVideoMCQDataset(
    # VideoBaseDataset
    ):

    # MD5 = '85bdd91f9b29a99354c23b97ab7c113c'
    SYS = ''

    FRAMES_TMPL_NOSUB = """
    These are the frames of a video. \
    Select the best answer to the following multiple-choice question based on the video. \
    Respond with only the letter (A, B, C, or D) of the correct option.
    """

    FRAMES_TMPL_SUB = """
    These are the frames of a video. \
    This video's subtitles are listed below:
    {}
    Select the best answer to the following multiple-choice question based on the video. \
    Respond with only the letter (A, B, C, or D) of the correct option.
    """

    TYPE = 'Video-MCQ'
    MODALITY = 'VIDEO'

    def __init__(self, dataset='LLaVA-Video-Multiple-Choice',
                #  use_subtitle=False, 
                nframe=0, fps=-1
                 ):
#         super().__init__(dataset=dataset, nframe=nframe, fps=fps)
#         self.use_subtitle = use_subtitle
        self.dataset_name = dataset
        ret = self.prepare_dataset(dataset)
        assert ret is not None
        
        lmu_root = LMUDataRoot()
        self.frame_root = osp.join(lmu_root, 'video_frame_root', dataset)
        
        os.makedirs(self.frame_root, exist_ok=True)
        self.frame_tmpl = 'frame-{}-of-{}.jpg'
        self.frame_tmpl_fps = 'frame-{}-of-{}-{}fps.jpg'

        self.data_root = ret['root']
        self.data_file = ret['data_file']
        self.data = load(self.data_file)
        if 'index' not in self.data:
            self.data['index'] = np.arange(len(self.data))

        assert 'question' in self.data and 'modality_path' in self.data
        videos = list(set(self.data['modality_path']))
        videos.sort()
        self.videos = videos
        # self.pack = pack
        self.nframe = nframe
        self.fps = fps
        if self.fps > 0 and self.nframe > 0:
            raise ValueError('fps and nframe should not be set at the same time')
        if self.fps <= 0 and self.nframe <= 0:
            raise ValueError('fps and nframe should be set at least one valid value')

    @classmethod
    def supported_datasets(cls):
        return ['LLaVA-Video-Multiple-Choice']

    def prepare_dataset(self, dataset_name='LLaVA-Video-Multiple-Choice',
                        # repo_id='llava-hf/LLaVA-NeXT-Video-7B-32K-hf'):
                        ):

        # def check_integrity(pth):
        # cache_path = get_cache_path(repo_id)
        # print("cache_path", cache_path)
        # sys.exit()
#         if cache_path is not None and check_integrity(cache_path):
#             dataset_path = cache_path
#         else:

#             def unzip_hf_zip(pth):
#             def generate_tsv(pth):
#             unzip_hf_zip(dataset_path)
#             generate_tsv(dataset_path)

        # dataset_path = repo_id
        dataset_path = "/path/to/home/LMUData/"
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        # data_file = osp.join(dataset_path, f'{dataset_name}_small.tsv')

        return dict(data_file=data_file, root=dataset_path)

    def frame_paths(self, video):
        # frame_root = osp.join(self.frame_root, video)
        # self.frame_root == '/path/to/home/LMUData/video_frame_root/LLaVA-Video-Multiple-Choice'
        # video.split("videos/")[-1].split(".mp4")[0] == 'academic_source/NextQA/1202/4295889026'
        # frame_root = osp.join(self.frame_root, video.split("videos")[-1].split(".mp4")[0])
        frame_root = os.path.join(self.frame_root, video.split("videos/")[-1].split(".mp4")[0])
        os.makedirs(frame_root, exist_ok=True) # the '/srv/muse-lab/datasets/LLaVA-Video/videos/academic_source/NextQA/1202/4295889026.mp4' exist already
        return [osp.join(frame_root, self.frame_tmpl.format(i, self.nframe)) for i in range(1, self.nframe + 1)]

    def save_video_frames(self, video, video_llm=False):

        # vid_path = osp.join(self.data_root, 'video', video + '.mp4')
        vid_path = video
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth) and not video_llm:
                    im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self.data)
            line = self.data.iloc[line]
        
        # to avoid SettingWithCopyWarning: 
        # A value is trying to be set on a copy of a slice from a DataFrame
        # See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
        # line['question'] += '\n' + '\n'.join(eval(line['candidates']))
        line = line.copy()

        # frames, indices, video_info = self.save_video_frames(line['modality_path'], video_llm)
        frames, _, _ = self.save_video_frames(line['modality_path'], video_llm)

#         if self.use_subtitle and os.path.exists(osp.join(self.data_root, line['subtitle_path'])):
#         else:

        message = [dict(type='text', value=self.SYS)]
        # message = [dict(type='text', text=line['question'])]
        if video_llm:
            # message.append(dict(type='video', value=osp.join(self.data_root, 'video', line['video'] + '.mp4')))
            message.append(dict(type='video', value=osp.join(self.data_root, 'video', line['modality_path'])))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))

#         text_prompt = self.FRAMES_TMPL_NOSUB if not self.use_subtitle else self.FRAMES_TMPL_SUB.format(subtitles)
#         message.append(dict(type='text', value=text_prompt))
#         line['question'] += '\n' + '\n'.join(eval(line['candidates']))
        prompt = 'Question: {}\nAnswer: '.format(line['question'])
        message.append(dict(type='text', value=prompt))
        return message
    
    def compare_response_n_label(self, response, label):
        if response in [" A ", "A ", " A", "A"] and label in ["0", "A", "A."]:
            return True
        elif response  in [" B ", "B ", " B", "B"] and label in ["1", "B", "B."]:
            return True
        elif response  in [" C ", "C ", " C", "C"] and label in ["2", "C", "C."]:
            return True
        elif response  in [" D ", "D ", " D", "D"] and label in ["3", "D", "D."]:
            return True
        else:
            return False

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        # from .utils.videomme import get_dimension_rating, extract_characters_regex, extract_option

        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'

        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')
            assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']

            if model == 'exact_matching':
                model = None
#             elif gpt_key_set():
#                 model = build_judge(**judge_kwargs)
#                 if not model.working():
#                     warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
#                     warnings.warn(DEBUG_MESSAGE)
#                     model = None
            else:
                warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
                model = None
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                # real_label = data.loc[data['index'] == idx, 'answer'].values[0]
                real_label = str(data.loc[data['index'] == idx, 'output'].values[0])
                model_prediction = str(data.loc[data['index'] == idx, 'prediction'].values[0])

#                 if extract_characters_regex(model_prediction) == '':
#                     extract_pred = extract_option(
#                         model,
#                         data.loc[data['index'] == idx].to_dict(orient='records')[0],
#                         'Video-MME'
#                     )
#                     data.loc[idx, 'score'] = int(extract_pred == real_label)
#                 else:
#                     data.loc[idx, 'score'] = int(extract_characters_regex(model_prediction) == real_label)

                data.loc[idx, 'score'] = int(self.compare_response_n_label(self, model_prediction, real_label))
            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

#         rating = get_dimension_rating(score_file)
#         dump(rating, tgt_file)
#         return rating
        return None
