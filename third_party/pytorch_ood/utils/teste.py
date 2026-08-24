from pytorch_ood.utils import *

a = Matriz_confusao_osr([1,2,3,4,5,6,7,8,9,0],
                        [0,1,2,3,4,5,6,7,8,-1],
                        [0,1,2,3,4,5,6,7,8,9],
                        [0],
                        col_labels=[9,0,1,2,3,4,5,6,7,8])
a.computa_matriz()
a.exibe_matriz()