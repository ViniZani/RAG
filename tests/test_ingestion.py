# flake8: noqa
# type: none
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter, PythonCodeTextSplitter
from src.student.ingestion import chunk_text, chunk_python_code

# (Cole a função chunk_text aqui em cima)

def test_code_blocks_preservation_in_large_section():
    # 1. Definimos os blocos de código com as crases do markdown inclusas
    bloco_1 = '```bash\necho "BLOCO_UNICO_1_START"\necho "BLOCO_UNICO_1_END"\n```'
    bloco_2 = '```bash\necho "BLOCO_UNICO_2_START"\necho "BLOCO_UNICO_2_END"\n```'
    
    # Parágrafos longos (~1.200 caracteres cada) para forçar o splitter a cortar a seção
    paragrafo_longo_1 = ( "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20 + "\n\n" )
    paragrafo_longo_2 = ( "Sed do eiusmod tempor incididunt ut labore et dolore. " * 20 + "\n\n" )
    paragrafo_longo_3 = ( "Ut enim ad minim veniam, quis nostrud exercitation. " * 20 + "\n\n" )

    # 2. Montamos UMA ÚNICA SEÇÃO com mais de 4.000 caracteres 
    texto_grande = ( "# Seção Única Gigante\n\n" + paragrafo_longo_1 + bloco_1 + "\n\n" + paragrafo_longo_2 + bloco_2 + "\n\n" + paragrafo_longo_3 )
    print(f"📏 Tamanho total do documento de teste: {len(texto_grande)} caracteres.")

    # 3. Executamos o chunking com max_chunk_size = 1500
    max_chunk_size = 1500 
    chunks = chunk_text(texto_grande, max_chunk_size=max_chunk_size)
    print(f"📦 Total de chunks gerados: {len(chunks)}\n")

    # --------------------------------------------------------- 
    # TESTE 1: Validação do start_index (Matemática do Offset) 
    # ---------------------------------------------------------
    for idx, doc in enumerate(chunks):
        start = doc.metadata["start_index"]
        content = doc.page_content
        slice_original = texto_grande[start : start + len(content)]
        
        assert slice_original == content, f"❌ Erro de offset no Chunk #{idx + 1} (start_index={start})!"
        
    print("✅ TESTE 1 PASSOU: Todos os start_indices estão 100% alinhados!")




    # --------------------------------------------------------- 
    # TESTE 2: Validação de Integridade dos Blocos de Código 
    # ---------------------------------------------------------
    blocos_para_testar = [("Bloco 1", bloco_1), ("Bloco 2", bloco_2)]
    
    for nome, bloco in blocos_para_testar:
        # Verifica se o bloco inteiro está contido na íntegra em pelo menos UM dos chunks
        bloco_intacto = any(bloco in doc.page_content for doc in chunks)
        assert bloco_intacto, f"❌ FALHA: O {nome} foi fragmentado ou cortado entre chunks!\nBloco esperado:\n{bloco}"
        print(f"✅ TESTE 2 PASSOU: {nome} foi preservado inteiro dentro de um único chunk!")


# =============================================================
# PythonCode Parser Text
# =============================================================

def test_python_code_preservation_in_large_file():
    # 1. Definimos funções e classes Python para simular um arquivo real
    funcao_1 = (
        'def calcular_soma(a: int, b: int) -> int:\n'
        '    """Calcula a soma de dois números inteiros."""\n'
        '    resultado = a + b\n'
        '    return resultado\n\n'
    )
    
    classe_1 = (
        'class ProcessadorDados:\n'
        '    """Classe responsável por processar os dados do sistema."""\n'
        '    def __init__(self, token: str):\n'
        '        self.token = token\n\n'
        '    def processar(self, dados: list) -> list:\n'
        '        return [d.strip() for d in dados if d]\n\n'
    )
    
    funcao_2 = (
        'def executar_pipeline() -> None:\n'
        '    """Executa o pipeline principal de ingestão."""\n'
        '    print("Iniciando pipeline...")\n'
        '    obj = ProcessadorDados("abc-123")\n'
        '    res = obj.processar(["  teste  ", "", "dados"])\n'
        '    print(res)\n'
    )

    # Parágrafos/comentários longos para forçar o splitter a quebrar o arquivo
    comentario_longo_1 = "# " + ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 15) + "\n\n"
    comentario_longo_2 = "# " + ("Sed do eiusmod tempor incididunt ut labore et dolore. " * 15) + "\n\n"

    # 2. Montamos UMA ÚNICA STRING de código Python grande
    codigo_grande = (
        '# Arquivo Python de Teste Gigante\n\n'
        + comentario_longo_1
        + funcao_1
        + comentario_longo_2
        + classe_1
        + funcao_2
    )
    print(f"📏 Tamanho total do arquivo Python de teste: {len(codigo_grande)} caracteres.")

    # 3. Executamos o chunking com um max_chunk_size restritivo
    max_chunk_size = 600
    chunks = chunk_python_code(codigo_grande, max_chunk_size=max_chunk_size)
    print(f"📦 Total de chunks gerados: {len(chunks)}\n")

    # ---------------------------------------------------------
    # TESTE 1: Validação do start_index (Matemática do Offset)
    # ---------------------------------------------------------
    for idx, doc in enumerate(chunks):
        start = doc.metadata.get("start_index", 0)
        content = doc.page_content
        slice_original = codigo_grande[start : start + len(content)]
        
        assert slice_original == content, (
            f"❌ Erro de offset no Chunk #{idx + 1} (start_index={start})!\n"
            f"Esperado: {repr(content[:30])}...\n"
            f"Encontrado: {repr(slice_original[:30])}..."
        )
        
    print("✅ TESTE 1 PASSOU: Todos os start_indices do Python estão 100% alinhados!")

    # ---------------------------------------------------------
    # TESTE 2: Validação de Integridade de Blocos Lógicos (Funções/Classes)
    # ---------------------------------------------------------
    estruturas_para_testar = [
        ("Função calcular_soma", funcao_1.strip()),
        ("Classe ProcessadorDados", classe_1.strip()),
        ("Função executar_pipeline", funcao_2.strip())
    ]
    
    for nome, estrutura in estruturas_para_testar:
        # Verifica se a estrutura inteira está contida na íntegra em pelo menos UM dos chunks
        estrutura_intacta = any(estrutura in doc.page_content for doc in chunks)
        assert estrutura_intacta, (
            f"❌ FALHA: A {nome} foi fragmentada ou cortada entre chunks!"
        )
        print(f"✅ TESTE 2 PASSOU: {nome} foi preservada inteira dentro de um único chunk!")


if __name__ == "__main__":
    while True:
        print("Select your choice:")
        data = input("1- Test .py parser\n2- Test .md parser\n")
        if data == "1":
            test_python_code_preservation_in_large_file()
        if data == "2":
            test_code_blocks_preservation_in_large_section()
        else:
            break
