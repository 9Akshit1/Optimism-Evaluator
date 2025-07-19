# optimism_runner.py
# External script to run the OptimismAnalyzer

from datetime import datetime, timedelta
import sys
import os

# Import your optimism analyzer module
# Assuming your code is saved as 'optimism_analyzer.py'
from optimism_analyzer import OptimismAnalyzer, TextData, create_sample_data

def run_optimism_analysis(text_data_list):
    """
    Run optimism analysis on provided text data
    
    Args:
        text_data_list: List of TextData objects or list of thread lists
    
    Returns:
        Dictionary containing analysis results
    """
    
    # Initialize the analyzer
    print("Initializing Optimism Analyzer...")
    analyzer = OptimismAnalyzer()
    print("✓ Analyzer initialized successfully")
    
    # Determine if we have single thread or multiple threads
    if isinstance(text_data_list[0], list):
        # Multiple threads
        print(f"Processing {len(text_data_list)} threads...")
        results = analyzer.analyze_batch(text_data_list)
        
        # Generate summaries for each thread
        thread_summaries = {}
        for thread_id, thread_results in results.items():
            thread_summaries[thread_id] = analyzer.get_thread_summary(thread_results)
        
        return {
            'individual_results': results,
            'thread_summaries': thread_summaries,
            'analysis_type': 'batch'
        }
    else:
        # Single thread
        print("Processing single thread...")
        results = analyzer.analyze_thread(text_data_list)
        summary = analyzer.get_thread_summary(results)
        
        return {
            'individual_results': results,
            'thread_summary': summary,
            'analysis_type': 'single'
        }

def create_custom_text_data(texts, user_ids=None, thread_id="custom_thread"):
    """
    Helper function to create TextData objects from simple text inputs
    
    Args:
        texts: List of text strings
        user_ids: List of user IDs (optional)
        thread_id: Thread identifier
    
    Returns:
        List of TextData objects
    """
    base_time = datetime.now()
    text_data_list = []
    
    for i, text in enumerate(texts):
        user_id = user_ids[i] if user_ids and i < len(user_ids) else f"user_{i+1}"
        
        text_data = TextData(
            text=text,
            timestamp=base_time - timedelta(minutes=i*10),  # 10 minutes apart
            thread_id=thread_id,
            user_id=user_id,
            upvotes=0,  # Default values
            downvotes=0
        )
        text_data_list.append(text_data)
    
    return text_data_list

def print_analysis_results(results):
    """Print formatted analysis results"""
    
    print("\n" + "="*60)
    print("OPTIMISM ANALYSIS RESULTS")
    print("="*60)
    
    if results['analysis_type'] == 'single':
        # Single thread results
        summary = results['thread_summary']
        individual = results['individual_results']
        
        print(f"Thread Optimism Score: {summary['mean_optimism']:.3f}/1.0")
        print(f"Optimism Trend: {summary['optimism_trend']:.3f}")
        print(f"Score Range: {summary['min_optimism']:.3f} - {summary['max_optimism']:.3f}")
        
        print("\nIndividual Post Scores:")
        for i, features in enumerate(individual):
            print(f"  Post {i+1}: {features.optimism_score:.3f}")
        
    else:
        # Multiple threads results
        summaries = results['thread_summaries']
        
        print("Thread Summary:")
        for thread_id, summary in summaries.items():
            print(f"\n{thread_id}:")
            print(f"  Optimism Score: {summary['mean_optimism']:.3f}/1.0")
            print(f"  Trend: {summary['optimism_trend']:.3f}")
            print(f"  Volatility: {summary['std_optimism']:.3f}")

def example_usage():
    """Demonstrate different ways to use the analyzer"""
    
    print("EXAMPLE 1: Using sample data")
    print("-" * 40)
    
    # Use built-in sample data
    sample_threads = create_sample_data()
    results = run_optimism_analysis(sample_threads)
    print_analysis_results(results)
    
    print("\n\nEXAMPLE 2: Custom text analysis")
    print("-" * 40)
    
    # Create custom text data for intern evaluation
    intern_communications = [
        "I'm really excited about this project! I think we can definitely make significant improvements to the user interface. I've been researching some innovative approaches that could solve the current usability issues.",
        
        "The current system has some challenges, but I believe we can work through them systematically. I'd love to collaborate with the team to find creative solutions.",
        
        "This is quite difficult and I'm not sure if we can meet the deadline. The requirements seem unclear and there are too many obstacles."
    ]
    
    custom_data = create_custom_text_data(
        intern_communications, 
        user_ids=["intern_A", "intern_B", "intern_C"],
        thread_id="intern_evaluation"
    )
    
    custom_results = run_optimism_analysis(custom_data)
    print_analysis_results(custom_results)
    
    # Detailed breakdown for intern evaluation
    print("\nINTERN EVALUATION BREAKDOWN:")
    print("-" * 40)
    
    for i, (comm, features) in enumerate(zip(intern_communications, custom_results['individual_results'])):
        print(f"\nIntern {chr(65+i)} Analysis:")
        print(f"  Text: '{comm[:80]}...'")
        print(f"  Optimism Score: {features.optimism_score:.3f}")
        print(f"  Future Orientation: {features.future_score:.3f}")
        print(f"  Action Tendency: {features.action_density:.3f}")
        print(f"  Enthusiasm Level: {features.enthusiasm:.3f}")
        print(f"  Certainty: {features.certainty:.3f}")

def main():
    """Main execution function"""
    
    print("Optimism Analyzer Runner")
    print("=" * 50)
    
    try:
        # Run example analysis
        example_usage()
        
        print("\n\n" + "="*60)
        print("Analysis completed successfully!")
        print("="*60)
        
        # Option to analyze custom input
        print("\nTo analyze your own text data:")
        print("1. Modify the 'intern_communications' list in example_usage()")
        print("2. Or create your own TextData objects")
        print("3. Call run_optimism_analysis() with your data")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Make sure 'optimism_analyzer.py' is in the same directory")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)