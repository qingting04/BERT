from hanlp.common.dataset import SortingSamplerBuilder
from hanlp.common.transform import NormalizeCharacter
from hanlp.components.mtl.multi_task_learning import MultiTaskLearning
from hanlp.components.mtl.tasks.tok.tag_tok import TaggingTokenization
from hanlp.layers.embeddings.contextual_word_embedding import ContextualWordEmbedding
from hanlp.utils.lang.zh.char_table import HANLP_CHAR_TABLE_JSON
from hanlp.utils.log_util import cprint

# 只配置分词任务
tasks = {
    'tok': TaggingTokenization(
        '199801-train.txt',
        '199801-dev.txt',
        '199801-test.txt',
        SortingSamplerBuilder(batch_size=32),
        max_seq_len=510,
        hard_constraint=True,
        char_level=True,
        tagging_scheme='BMES',
        lr=1e-3,
        crf=True,
        transform=NormalizeCharacter(HANLP_CHAR_TABLE_JSON, 'token'),
    )
}

mtl = MultiTaskLearning()
save_dir = './tok_model'
mtl.fit(
    ContextualWordEmbedding('token',
                            "./pre_model/chinese-electra",
                            average_subwords=True,
                            max_sequence_length=512,
                            word_dropout=.1),
    tasks,
    save_dir,
    10,  # 训练周期，根据需要调整
    lr=5e-5,  # 学习率，根据需要调整
    encoder_lr=2e-5,  # 编码器学习率，根据需要调整
    grad_norm=1,
    gradient_accumulation=1,
    eval_trn=False,
)
cprint(f'Model saved in [cyan]{save_dir}[/cyan]')
mtl.evaluate(save_dir)
mtl.load(save_dir)
print(mtl('华纳音乐旗下的新垣结衣在12月21日于日本武道馆举办歌手出道活动'))
