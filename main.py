import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# Load environment variables
load_dotenv()

# 1. NEW LLM INITIALIZATION
# CrewAI now uses its own LLM wrapper to connect to Groq
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY")
)

# 2. FILE NAME UPDATE
# Read our dummy files (updated to match your new file name)
with open('legacy_math.cu', 'r') as file:
    cuda_code = file.read()

with open('amd_hip_hop_docs.txt', 'r') as file:
    hip_docs = file.read()

# ==========================================
# 3. DEFINE THE AGENTS
# ==========================================
analyzer = Agent(
    role='Senior CUDA Architect',
    goal='Analyze CUDA C++ files and identify all NVIDIA-specific APIs that need migration.',
    backstory='You are an expert in parallel computing, specifically NVIDIA architectures. You excel at finding legacy CUDA calls.',
    llm=llm,
    verbose=True
)

translator = Agent(
    role='AMD ROCm Migration Specialist',
    goal='Translate CUDA code to AMD HIP C++ code accurately using provided documentation.',
    backstory='You are an AMD engineer who specializes in porting legacy NVIDIA code to the ROCm ecosystem. You strictly follow official AMD documentation.',
    llm=llm,
    verbose=True
)

reviewer = Agent(
    role='Staff QA Engineer',
    goal='Review translated HIP code, format it beautifully, and generate a markdown migration report.',
    backstory='You have a meticulous eye for C++ syntax. You ensure developers understand exactly what changed in their codebase.',
    llm=llm,
    verbose=True
)

# ==========================================
# 4. DEFINE THE TASKS
# ==========================================
analyze_task = Task(
    description=f'Read the following CUDA code and list all NVIDIA-specific functions that require translation.\n\nCode:\n{cuda_code}',
    expected_output='A bulleted list of CUDA APIs found in the code.',
    agent=analyzer
)

translate_task = Task(
    description=f'Take the analysis from the previous step. Rewrite the original CUDA code into AMD HIP C++ syntax. Use this documentation as your absolute source of truth:\n\n{hip_docs}\n\nOriginal Code:\n{cuda_code}',
    expected_output='The fully translated C++ code using HIP syntax.',
    agent=translator
)

review_task = Task(
    description='Review the translated code from the previous step. Ensure it includes <hip/hip_runtime.h>. Create a final output that contains:\n1. A brief summary of what was changed.\n2. The final translated code block.',
    expected_output='A clean Markdown formatted report containing the summary and the final code.',
    agent=reviewer,
    output_file='Migration_Report.md' 
)

# ==========================================
# 5. KICKOFF THE SWARM
# ==========================================
migration_crew = Crew(
    agents=[analyzer, translator, reviewer],
    tasks=[analyze_task, translate_task, review_task],
    process=Process.sequential 
)

print("Starting the HIP-Hop Migration Swarm...\n")
result = migration_crew.kickoff()
print("\nMigration Complete! Check the Migration_Report.md file.")