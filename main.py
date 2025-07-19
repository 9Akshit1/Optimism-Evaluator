import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import re
import statistics
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import math
from collections import Counter, defaultdict
from textstat import flesch_reading_ease, flesch_kincaid_grade
import syllables
from datetime import timezone

import warnings
warnings.filterwarnings('ignore')

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("Please install vaderSentiment: pip install vaderSentiment")

try:
    from transformers import pipeline
except ImportError:
    print("Please install transformers: pip install transformers torch")

@dataclass
class TextData:
    text: str
    timestamp: datetime
    thread_id: str
    user_id: str
    parent_id: Optional[str] = None
    upvotes: int = 0
    downvotes: int = 0

@dataclass
class OptimismFeatures:
    vader_pos: float
    vader_neg: float
    vader_neu: float
    vader_comp: float
    roberta_pos: float
    roberta_neg: float
    roberta_neu: float
    response_speed: float
    time_momentum: float
    peak_align: float
    future_score: float
    action_density: float
    certainty: float
    enthusiasm: float
    engagement: float
    thread_pos: float
    upvote_ratio: float
    optimism_score: float

class OptimismAnalyzer:
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        try:
            self.roberta = pipeline("sentiment-analysis", 
                                  model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                  return_all_scores=True)
        except Exception as e:
            print(f"RoBERTa model loading failed: {e}")
            self.roberta = None
        
        self.future_words = {
            'will', 'going', 'future', 'tomorrow', 'next', 'soon', 'upcoming', 
            'potential', 'opportunity', 'growth', 'improvement', 'progress',
            'advance', 'develop', 'evolve', 'expand', 'increase', 'rise'
        }
        
        self.action_words = {
            'do', 'make', 'create', 'build', 'achieve', 'accomplish', 'succeed',
            'win', 'overcome', 'solve', 'fix', 'improve', 'enhance', 'boost',
            'start', 'begin', 'launch', 'implement', 'execute', 'deliver'
        }
        
        self.certainty_words = {
            'definitely', 'certainly', 'absolutely', 'surely', 'clearly',
            'obviously', 'undoubtedly', 'confident', 'sure', 'convinced',
            'believe', 'know', 'expect', 'anticipate', 'guarantee'
        }
        
        self.enthusiasm_markers = {
            '!', '!!', '!!!', 'amazing', 'awesome', 'fantastic', 'incredible', 'wonderful',
            'excellent', 'brilliant', 'outstanding', 'superb', 'great'
        }
        
        self.scaler = MinMaxScaler()
        self.optimism_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def extract_time_features(self, text_data: List[TextData]) -> Dict[str, float]:
        if len(text_data) < 2:
            return {'response_speed': 0.5, 'time_momentum': 0.5, 'peak_align': 0.5}
        
        sorted_data = sorted(text_data, key=lambda x: x.timestamp)
        
        response_times = []
        for i in range(1, len(sorted_data)):
            time_diff = (sorted_data[i].timestamp - sorted_data[i-1].timestamp).total_seconds()
            response_times.append(time_diff)
        
        avg_response_time = np.mean(response_times) if response_times else 3600
        response_speed = max(0, min(1, 1 - (avg_response_time / 86400)))
        
        time_windows = self.create_time_windows(sorted_data)
        activity_trend = self.calc_activity_trend(time_windows)
        time_momentum = max(0, min(1, (activity_trend + 1) / 2))
        
        peak_align = self.calc_peak_alignment(time_windows)
        
        return {
            'response_speed': response_speed,
            'time_momentum': time_momentum,
            'peak_align': peak_align
        }
    
    def create_time_windows(self, sorted_data: List[TextData], window_size: int = 3600) -> List[int]:
        if not sorted_data:
            return []
        
        start_time = sorted_data[0].timestamp
        end_time = sorted_data[-1].timestamp
        total_seconds = (end_time - start_time).total_seconds()
        
        num_windows = max(1, int(total_seconds / window_size))
        windows = [0] * num_windows
        
        for data in sorted_data:
            window_index = int((data.timestamp - start_time).total_seconds() / window_size)
            window_index = min(window_index, num_windows - 1)
            windows[window_index] += 1
        
        return windows
    
    def calc_activity_trend(self, windows: List[int]) -> float:
        if len(windows) < 2:
            return 0
        
        x = np.arange(len(windows))
        y = np.array(windows)
        
        if np.std(x) == 0:
            return 0
        
        correlation = np.corrcoef(x, y)[0, 1]
        return correlation if not np.isnan(correlation) else 0
    
    def calc_peak_alignment(self, windows: List[int]) -> float:
        if len(windows) < 3:
            return 0.5
        
        peaks = []
        for i in range(1, len(windows) - 1):
            if windows[i] > windows[i-1] and windows[i] > windows[i+1]:
                peaks.append(i)
        
        if len(peaks) < 2:
            return 0.5
        
        peak_intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
        if not peak_intervals:
            return 0.5
        
        interval_std = np.std(peak_intervals)
        interval_mean = np.mean(peak_intervals)
        
        if interval_mean == 0:
            return 0.5
        
        regularity = 1 - min(1, interval_std / interval_mean)
        return regularity
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        total_words = len(words)
        
        if total_words == 0:
            return {
                'future_score': 0,
                'action_density': 0,
                'certainty': 0,
                'enthusiasm': 0
            }
        
        future_count = sum(1 for word in words if word in self.future_words)
        future_score = future_count / total_words
        
        action_count = sum(1 for word in words if word in self.action_words)
        action_density = action_count / total_words
        
        certainty_count = sum(1 for word in words if word in self.certainty_words)
        certainty = certainty_count / total_words
        
        enthusiasm_count = 0
        for marker in self.enthusiasm_markers:
            if marker in text:
                enthusiasm_count += text.count(marker)
        
        enthusiasm = min(1.0, enthusiasm_count / max(1, len(text) / 100))
        
        return {
            'future_score': future_score,
            'action_density': action_density,
            'certainty': certainty,
            'enthusiasm': enthusiasm
        }
    
    def extract_behavioral_features(self, data: TextData, thread_data: List[TextData]) -> Dict[str, float]:
        engagement = min(1.0, len(data.text) / 1000)
        
        thread_positions = sorted([d.timestamp for d in thread_data])
        if len(thread_positions) > 1:
            position_index = thread_positions.index(data.timestamp)
            thread_pos = 1 - (position_index / len(thread_positions))
        else:
            thread_pos = 1.0
        
        total_votes = data.upvotes + data.downvotes
        if total_votes > 0:
            upvote_ratio = data.upvotes / total_votes
        else:
            upvote_ratio = 0.5
        
        return {
            'engagement': engagement,
            'thread_pos': thread_pos,
            'upvote_ratio': upvote_ratio
        }
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        vader_scores = self.vader.polarity_scores(text)
        
        roberta_scores = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
        if self.roberta:
            try:
                roberta_result = self.roberta(text)[0]
                roberta_scores = {score['label'].lower(): score['score'] for score in roberta_result}
                if 'label_2' in roberta_scores:
                    roberta_scores['positive'] = roberta_scores.get('label_2', 0)
                if 'label_0' in roberta_scores:
                    roberta_scores['negative'] = roberta_scores.get('label_0', 0)
                if 'label_1' in roberta_scores:
                    roberta_scores['neutral'] = roberta_scores.get('label_1', 0)
            except Exception as e:
                pass
        
        return {
            'vader_pos': vader_scores['pos'],
            'vader_neg': vader_scores['neg'],
            'vader_neu': vader_scores['neu'],
            'vader_comp': vader_scores['compound'],
            'roberta_pos': roberta_scores.get('positive', 0.33),
            'roberta_neg': roberta_scores.get('negative', 0.33),
            'roberta_neu': roberta_scores.get('neutral', 0.34)
        }

    def extract_creative_optimism_features(self, text: str, data: TextData, thread_data: List[TextData]) -> Dict[str, float]:
        """
        Extract truly creative and novel optimism indicators that go beyond traditional approaches
        """
        features = {}
        
        # 1. COGNITIVE LOAD ANALYSIS - Optimistic people think more complexly about positive futures
        features.update(self._analyze_cognitive_complexity(text))
        
        # 2. TEMPORAL LINGUISTIC PATTERNS - How people structure time in language reveals optimism
        features.update(self._analyze_temporal_linguistics(text))
        
        # 3. SOCIAL CONTAGION PATTERNS - How optimism spreads through conversation chains
        features.update(self._analyze_social_contagion(data, thread_data))
        
        # 4. LINGUISTIC ENERGY SIGNATURE - Optimism has a unique "energy fingerprint" in language
        features.update(self._analyze_linguistic_energy(text))
        
        # 5. SOLUTION-FOCUSED LANGUAGE - Optimists naturally frame problems as solvable
        features.update(self._analyze_solution_framing(text))
        
        # 6. COLLECTIVE PRONOUN DYNAMICS - Optimists use "we/us" vs "I/me" differently
        features.update(self._analyze_pronoun_dynamics(text))
        
        # 7. SEMANTIC BRIDGING - Optimists connect seemingly unrelated positive concepts
        features.update(self._analyze_semantic_bridging(text))
        
        # 8. TEMPORAL RESPONSE CLUSTERING - Optimistic discussions create unique timing patterns
        features.update(self._analyze_response_clustering(data, thread_data))
        
        return features

    def _analyze_cognitive_complexity(self, text: str) -> Dict[str, float]:
        """
        NOVEL INSIGHT: Optimistic people engage in more complex thinking about positive outcomes.
        They use more subordinate clauses, conditional statements, and layered reasoning.
        """
        # Count complex sentence structures
        subordinate_markers = ['because', 'since', 'although', 'while', 'whereas', 'if', 'unless', 'when', 'where']
        conditional_markers = ['if', 'unless', 'provided', 'assuming', 'suppose', 'imagine']
        causal_chains = ['therefore', 'thus', 'consequently', 'as a result', 'leading to', 'which means']
        
        words = text.lower().split()
        total_words = len(words)
        
        if total_words == 0:
            return {'cognitive_complexity': 0, 'conditional_thinking': 0, 'causal_reasoning': 0}
        
        # Measure subordinate clause density
        subordinate_count = sum(1 for marker in subordinate_markers if marker in text.lower())
        subordinate_density = subordinate_count / max(1, total_words / 10)  # Per 10 words
        
        # Measure conditional thinking (optimists explore "what if" scenarios)
        conditional_count = sum(1 for marker in conditional_markers if marker in text.lower())
        conditional_density = conditional_count / max(1, total_words / 10)
        
        # Measure causal reasoning chains
        causal_count = sum(1 for marker in causal_chains if marker in text.lower())
        causal_density = causal_count / max(1, total_words / 10)
        
        # Complex punctuation patterns (optimists use more varied punctuation)
        punctuation_variety = len(set(char for char in text if char in '!?;:()[]{}"-')) / max(1, len(text) / 100)
        
        # Reading complexity (optimists write more sophisticatedly about positive topics)
        try:
            reading_ease = flesch_reading_ease(text)
            complexity_score = max(0, (100 - reading_ease) / 100)  # Higher complexity = lower ease
        except:
            complexity_score = 0.5
        
        overall_complexity = (subordinate_density + conditional_density + causal_density + 
                            punctuation_variety + complexity_score) / 5
        
        return {
            'cognitive_complexity': min(1.0, overall_complexity),
            'conditional_thinking': min(1.0, conditional_density),
            'causal_reasoning': min(1.0, causal_density)
        }

    def _analyze_temporal_linguistics(self, text: str) -> Dict[str, float]:
        """
        NOVEL INSIGHT: Optimists have unique patterns in how they structure time in language.
        They create "temporal bridges" connecting present actions to future outcomes.
        """
        # Temporal bridging phrases (connecting present to future)
        bridge_phrases = [
            'will lead to', 'going to result in', 'setting up for', 'paving the way',
            'building toward', 'working toward', 'moving toward', 'progressing to',
            'evolving into', 'developing into', 'growing into', 'becoming'
        ]
        
        # Future-conditional constructions (optimistic scenario building)
        future_conditionals = [
            'when we', 'once we', 'after we', 'as soon as', 'by the time',
            'will be able to', 'going to be', 'will have', 'will see'
        ]
        
        # Progressive momentum indicators
        momentum_words = [
            'accelerating', 'gaining', 'building', 'growing', 'expanding',
            'increasing', 'improving', 'advancing', 'progressing', 'developing'
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        if total_words == 0:
            return {'temporal_bridging': 0, 'future_conditionals': 0, 'momentum_language': 0}
        
        # Count temporal bridges
        bridge_count = sum(1 for phrase in bridge_phrases if phrase in text_lower)
        temporal_bridging = bridge_count / max(1, total_words / 20)
        
        # Count future conditionals
        conditional_count = sum(1 for phrase in future_conditionals if phrase in text_lower)
        future_conditionals_score = conditional_count / max(1, total_words / 20)
        
        # Count momentum language
        momentum_count = sum(1 for word in momentum_words if word in words)
        momentum_score = momentum_count / max(1, total_words / 10)
        
        # Temporal sequence complexity (optimists describe multi-step futures)
        sequence_markers = ['first', 'then', 'next', 'after that', 'finally', 'eventually']
        sequence_count = sum(1 for marker in sequence_markers if marker in text_lower)
        sequence_complexity = sequence_count / max(1, total_words / 15)
        
        return {
            'temporal_bridging': min(1.0, temporal_bridging),
            'future_conditionals': min(1.0, future_conditionals_score),
            'momentum_language': min(1.0, momentum_score),
            'sequence_complexity': min(1.0, sequence_complexity)
        }

    def _analyze_social_contagion(self, data: TextData, thread_data: List[TextData]) -> Dict[str, float]:
        """
        NOVEL INSIGHT: Optimism spreads through social networks in measurable patterns.
        Optimistic posts create "contagion cascades" that influence subsequent responses.
        """
        if len(thread_data) < 3:
            return {'contagion_coefficient': 0.5, 'cascade_trigger': 0.5, 'momentum_shift': 0.5}
        
        # Find position of current post in thread
        sorted_thread = sorted(thread_data, key=lambda x: x.timestamp)
        try:
            current_index = next(i for i, post in enumerate(sorted_thread) if post.timestamp == data.timestamp)
        except:
            current_index = len(sorted_thread) // 2
        
        # Analyze sentiment momentum before and after this post
        if current_index > 0 and current_index < len(sorted_thread) - 1:
            # Get sentiment of posts before and after
            before_posts = sorted_thread[max(0, current_index-2):current_index]
            after_posts = sorted_thread[current_index+1:min(len(sorted_thread), current_index+3)]
            
            # Quick sentiment scoring for surrounding posts
            before_sentiment = np.mean([self._quick_sentiment_score(post.text) for post in before_posts])
            after_sentiment = np.mean([self._quick_sentiment_score(post.text) for post in after_posts])
            current_sentiment = self._quick_sentiment_score(data.text)
            
            # Contagion coefficient: how much this post influences subsequent posts
            contagion_coefficient = max(0, min(1, (after_sentiment - before_sentiment) / 2 + 0.5))
            
            # Cascade trigger: does this post start a positive cascade?
            cascade_trigger = 1.0 if (current_sentiment > 0.6 and after_sentiment > before_sentiment + 0.2) else 0.0
            
            # Momentum shift: does this post change the conversation direction?
            momentum_shift = abs(after_sentiment - before_sentiment) if current_sentiment > 0.5 else 0
        else:
            contagion_coefficient = 0.5
            cascade_trigger = 0.5
            momentum_shift = 0.5
        
        # Response time analysis - optimistic posts get faster responses
        if current_index < len(sorted_thread) - 1:
            response_time = (sorted_thread[current_index + 1].timestamp - data.timestamp).total_seconds()
            quick_response_score = max(0, min(1, 1 - (response_time / 3600)))  # 1 hour max
        else:
            quick_response_score = 0.5
        
        return {
            'contagion_coefficient': contagion_coefficient,
            'cascade_trigger': cascade_trigger,
            'momentum_shift': min(1.0, momentum_shift),
            'response_magnetism': quick_response_score
        }

    def _quick_sentiment_score(self, text: str) -> float:
        """Quick sentiment scoring for contagion analysis"""
        positive_words = ['good', 'great', 'amazing', 'awesome', 'excellent', 'love', 'best', 'incredible']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing']
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        if pos_count + neg_count == 0:
            return 0.5
        
        return pos_count / (pos_count + neg_count)

    def _analyze_linguistic_energy(self, text: str) -> Dict[str, float]:
        """
        NOVEL INSIGHT: Optimism has a unique "energy signature" in language patterns.
        This includes rhythm, syllable patterns, and phonetic characteristics.
        """
        words = text.split()
        if not words:
            return {'rhythmic_energy': 0, 'syllabic_momentum': 0, 'phonetic_brightness': 0}
        
        # Rhythmic energy: variation in word and sentence lengths
        word_lengths = [len(word) for word in words]
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentence_lengths = [len(s.split()) for s in sentences if s]
        
        if len(word_lengths) > 1:
            word_rhythm = np.std(word_lengths) / np.mean(word_lengths) if np.mean(word_lengths) > 0 else 0
        else:
            word_rhythm = 0
        
        if len(sentence_lengths) > 1:
            sentence_rhythm = np.std(sentence_lengths) / np.mean(sentence_lengths) if np.mean(sentence_lengths) > 0 else 0
        else:
            sentence_rhythm = 0
        
        rhythmic_energy = (word_rhythm + sentence_rhythm) / 2
        
        # Syllabic momentum: optimists use more dynamic syllable patterns
        try:
            total_syllables = sum(syllables.estimate(word) for word in words if word.isalpha())
            avg_syllables = total_syllables / len(words) if words else 0
            syllabic_momentum = min(1.0, avg_syllables / 3)  # Normalize around 3 syllables average
        except:
            syllabic_momentum = 0.5
        
        # Phonetic brightness: certain sounds are associated with optimism
        bright_sounds = ['ee', 'ay', 'ai', 'ey', 'ie', 'y']  # High, bright vowel sounds
        dark_sounds = ['oo', 'ou', 'ow', 'au', 'aw']  # Low, dark vowel sounds
        
        text_lower = text.lower()
        bright_count = sum(text_lower.count(sound) for sound in bright_sounds)
        dark_count = sum(text_lower.count(sound) for sound in dark_sounds)
        
        if bright_count + dark_count > 0:
            phonetic_brightness = bright_count / (bright_count + dark_count)
        else:
            phonetic_brightness = 0.5
        
        return {
            'rhythmic_energy': min(1.0, rhythmic_energy),
            'syllabic_momentum': syllabic_momentum,
            'phonetic_brightness': phonetic_brightness
        }

    def _analyze_solution_framing(self, text: str) -> Dict[str, float]:
        """
        NOVEL INSIGHT: Optimists naturally frame challenges as solvable problems.
        They use "solution architecture" language patterns.
        """
        # Solution-oriented language patterns
        solution_starters = [
            'we can', 'we could', 'we should', 'let\'s', 'what if we',
            'one way to', 'another approach', 'alternative', 'solution',
            'fix', 'solve', 'address', 'tackle', 'handle', 'deal with'
        ]
        
        # Problem-to-solution bridges
        bridge_patterns = [
            'but we can', 'however we could', 'still possible', 'way around',
            'work through', 'figure out', 'find a way', 'make it work'
        ]
        
        # Collaborative solution language
        collaborative_words = [
            'together', 'team up', 'combine', 'join forces', 'work with',
            'pool resources', 'share', 'contribute', 'help each other'
        ]
        
        # Innovation/creativity indicators
        innovation_words = [
            'innovate', 'creative', 'think outside', 'new approach', 'fresh perspective',
            'breakthrough', 'game changer', 'revolutionary', 'disruptive'
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        if total_words == 0:
            return {'solution_orientation': 0, 'collaborative_framing': 0, 'innovation_mindset': 0}
        
        # Count solution-oriented language
        solution_count = sum(1 for pattern in solution_starters + bridge_patterns if pattern in text_lower)
        solution_orientation = solution_count / max(1, total_words / 15)
        
        # Count collaborative language
        collab_count = sum(1 for word in collaborative_words if word in text_lower)
        collaborative_framing = collab_count / max(1, total_words / 15)
        
        # Count innovation language
        innovation_count = sum(1 for word in innovation_words if word in text_lower)
        innovation_mindset = innovation_count / max(1, total_words / 15)
        
        # Problem acknowledgment + solution pattern
        problem_words = ['problem', 'issue', 'challenge', 'difficulty', 'obstacle']
        solution_words = ['solution', 'answer', 'fix', 'resolve', 'overcome']
        
        has_problem = any(word in text_lower for word in problem_words)
        has_solution = any(word in text_lower for word in solution_words)
        problem_solution_pattern = 1.0 if (has_problem and has_solution) else 0.5
        
        return {
            'solution_orientation': min(1.0, solution_orientation),
            'collaborative_framing': min(1.0, collaborative_framing),
            'innovation_mindset': min(1.0, innovation_mindset),
            'problem_solution_pattern': problem_solution_pattern
        }

    def _analyze_pronoun_dynamics(self, text: str) -> Dict[str, float]:
        """
        NOVEL INSIGHT: Optimists use pronouns differently - more "we/us" when discussing positive outcomes,
        more "I/me" when taking responsibility for action.
        """
        import re
        
        # Extract all pronouns with context
        collective_pronouns = ['we', 'us', 'our', 'ours', 'ourselves']
        individual_pronouns = ['i', 'me', 'my', 'mine', 'myself']
        
        words = text.lower().split()
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        
        # Count pronouns in positive vs negative contexts
        positive_context_words = ['can', 'will', 'great', 'amazing', 'success', 'win', 'achieve', 'accomplish']
        negative_context_words = ['can\'t', 'won\'t', 'bad', 'fail', 'problem', 'issue', 'difficult']
        
        collective_in_positive = 0
        individual_in_positive = 0
        total_collective = 0
        total_individual = 0
        
        for sentence in sentences:
            sentence_words = sentence.lower().split()
            has_positive = any(word in sentence_words for word in positive_context_words)
            has_negative = any(word in sentence_words for word in negative_context_words)
            
            for word in sentence_words:
                if word in collective_pronouns:
                    total_collective += 1
                    if has_positive and not has_negative:
                        collective_in_positive += 1
                elif word in individual_pronouns:
                    total_individual += 1
                    if has_positive and not has_negative:
                        individual_in_positive += 1
        
        # Calculate ratios
        if total_collective > 0:
            collective_optimism_ratio = collective_in_positive / total_collective
        else:
            collective_optimism_ratio = 0.5
        
        if total_individual > 0:
            individual_optimism_ratio = individual_in_positive / total_individual
        else:
            individual_optimism_ratio = 0.5
        
        # Collective vs individual balance in positive contexts
        total_positive_pronouns = collective_in_positive + individual_in_positive
        if total_positive_pronouns > 0:
            collective_positive_ratio = collective_in_positive / total_positive_pronouns
        else:
            collective_positive_ratio = 0.5
        
        return {
            'collective_optimism_ratio': collective_optimism_ratio,
            'individual_optimism_ratio': individual_optimism_ratio,
            'collective_positive_balance': collective_positive_ratio
        }

    def _analyze_semantic_bridging(self, text: str) -> Dict[str, float]:
        """
        NOVEL INSIGHT: Optimists create unexpected connections between positive concepts,
        building "semantic bridges" that reveal creative, hopeful thinking.
        """
        # Define semantic domains
        domains = {
            'technology': ['tech', 'digital', 'innovation', 'AI', 'future', 'smart', 'advanced'],
            'growth': ['grow', 'expand', 'increase', 'develop', 'improve', 'progress', 'evolve'],
            'community': ['together', 'community', 'team', 'collective', 'shared', 'united'],
            'success': ['success', 'achieve', 'accomplish', 'win', 'triumph', 'excel', 'thrive'],
            'creativity': ['creative', 'innovative', 'original', 'unique', 'artistic', 'imaginative'],
            'opportunity': ['opportunity', 'chance', 'potential', 'possibility', 'opening', 'prospect']
        }
        
        words = text.lower().split()
        
        # Find which domains are present
        active_domains = []
        for domain, domain_words in domains.items():
            if any(word in words for word in domain_words):
                active_domains.append(domain)
        
        # Calculate semantic bridging score
        num_domains = len(active_domains)
        if num_domains <= 1:
            bridging_score = 0
        else:
            # More domains connected = higher bridging
            bridging_score = min(1.0, (num_domains - 1) / 4)  # Normalize to 0-1
        
        # Connection strength: look for connecting words between domains
        connection_words = ['and', 'plus', 'with', 'through', 'via', 'by', 'using', 'combining']
        connection_count = sum(1 for word in connection_words if word in words)
        connection_strength = min(1.0, connection_count / max(1, len(words) / 20))
        
        # Metaphorical language (optimists use more metaphors)
        metaphor_indicators = ['like', 'as', 'similar to', 'reminds me of', 'just like', 'kind of like']
        metaphor_count = sum(1 for indicator in metaphor_indicators if indicator in text.lower())
        metaphorical_richness = min(1.0, metaphor_count / max(1, len(words) / 25))
        
        return {
            'semantic_bridging': bridging_score,
            'connection_strength': connection_strength,
            'metaphorical_richness': metaphorical_richness
        }

    def _analyze_response_clustering(self, data: TextData, thread_data: List[TextData]) -> Dict[str, float]:
        """
        NOVEL INSIGHT: Optimistic discussions create unique temporal clustering patterns.
        Optimism creates "gravity wells" that attract more responses.
        """
        if len(thread_data) < 5:
            return {'temporal_gravity': 0.5, 'cluster_density': 0.5, 'optimism_magnetism': 0.5}
        
        sorted_thread = sorted(thread_data, key=lambda x: x.timestamp)
        
        try:
            current_index = next(i for i, post in enumerate(sorted_thread) if post.timestamp == data.timestamp)
        except:
            return {'temporal_gravity': 0.5, 'cluster_density': 0.5, 'optimism_magnetism': 0.5}
        
        # Analyze clustering around this post
        window_size = 2  # Look 2 posts before and after
        start_idx = max(0, current_index - window_size)
        end_idx = min(len(sorted_thread), current_index + window_size + 1)
        
        window_posts = sorted_thread[start_idx:end_idx]
        
        if len(window_posts) < 3:
            return {'temporal_gravity': 0.5, 'cluster_density': 0.5, 'optimism_magnetism': 0.5}
        
        # Calculate time gaps between posts in window
        time_gaps = []
        for i in range(len(window_posts) - 1):
            gap = (window_posts[i + 1].timestamp - window_posts[i].timestamp).total_seconds()
            time_gaps.append(gap)
        
        # Temporal gravity: how much this post attracts nearby responses
        avg_gap = np.mean(time_gaps) if time_gaps else 3600
        temporal_gravity = max(0, min(1, 1 - (avg_gap / 7200)))  # 2 hours max
        
        # Cluster density: how compressed the responses are
        if len(time_gaps) > 1:
            gap_std = np.std(time_gaps)
            gap_mean = np.mean(time_gaps)
            if gap_mean > 0:
                cluster_density = max(0, min(1, 1 - (gap_std / gap_mean)))
            else:
                cluster_density = 0.5
        else:
            cluster_density = 0.5
        
        # Optimism magnetism: do positive posts attract more responses?
        current_sentiment = self._quick_sentiment_score(data.text)
        response_count = len([p for p in window_posts if p.timestamp > data.timestamp])
        max_possible_responses = min(3, len(sorted_thread) - current_index - 1)
        
        if max_possible_responses > 0:
            response_ratio = response_count / max_possible_responses
            optimism_magnetism = response_ratio * current_sentiment
        else:
            optimism_magnetism = 0.5
        
        return {
            'temporal_gravity': temporal_gravity,
            'cluster_density': cluster_density,
            'optimism_magnetism': min(1.0, optimism_magnetism)
        }
    
    def calc_optimism_score(self, features: Dict[str, float]) -> float:
        weights = {
            # Traditional sentiment features (30% total weight)
            'vader_positive': 0.10,
            'vader_compound': 0.08,
            'roberta_positive': 0.12,
            
            # Temporal features (20% total weight)
            'response_speed_score': 0.06,
            'temporal_momentum': 0.08,
            'peak_activity_alignment': 0.06,
            
            # Linguistic features (20% total weight)
            'future_orientation_score': 0.05,
            'action_word_density': 0.05,
            'certainty_score': 0.04,
            'enthusiasm_markers': 0.06,
            
            # Behavioral features (8% total weight)
            'engagement_quality': 0.03,
            'thread_position_score': 0.02,
            'upvote_ratio': 0.03,
            
            # CREATIVE OPTIMISM FEATURES (22% total weight)
            # Cognitive complexity
            'cognitive_complexity': 0.04,
            'conditional_thinking': 0.03,
            'causal_reasoning': 0.02,
            
            # Temporal linguistics
            'temporal_bridging': 0.03,
            'momentum_language': 0.02,
            'sequence_complexity': 0.02,
            
            # Social contagion
            'contagion_coefficient': 0.03,
            'response_magnetism': 0.02,
            
            # Linguistic energy
            'rhythmic_energy': 0.02,
            'phonetic_brightness': 0.01,
            
            # Solution framing
            'solution_orientation': 0.03,
            'innovation_mindset': 0.02,
            
            # Pronoun dynamics
            'collective_positive_balance': 0.02,
            
            # Semantic bridging
            'semantic_bridging': 0.02,
        }
        
        optimism_score = 0
        for feature, weight in weights.items():
            if feature in features:
                optimism_score += features[feature] * weight
        
        optimism_score = 1 / (1 + np.exp(-5 * (optimism_score - 0.5)))
        
        return min(1.0, max(0.0, optimism_score))
    
    def analyze_thread(self, thread_data: List[TextData]) -> List[OptimismFeatures]:
        if not thread_data:
            return []
        
        time_features = self.extract_time_features(thread_data)
        
        results = []
        
        for data in thread_data:
            sentiment_features = self.analyze_text_sentiment(data.text)
            linguistic_features = self.extract_linguistic_features(data.text)
            behavioral_features = self.extract_behavioral_features(data, thread_data)

            creative_features = self.extract_creative_optimism_features(data.text, data, thread_data)
            
            all_features = {
                **sentiment_features,
                **time_features,
                **linguistic_features,
                **behavioral_features,
                **creative_features
            }
            
            optimism_score = self.calc_optimism_score(all_features)
            
            features = OptimismFeatures(
                vader_pos=sentiment_features['vader_pos'],
                vader_neg=sentiment_features['vader_neg'],
                vader_neu=sentiment_features['vader_neu'],
                vader_comp=sentiment_features['vader_comp'],
                roberta_pos=sentiment_features['roberta_pos'],
                roberta_neg=sentiment_features['roberta_neg'],
                roberta_neu=sentiment_features['roberta_neu'],
                response_speed=time_features['response_speed'],
                time_momentum=time_features['time_momentum'],
                peak_align=time_features['peak_align'],
                future_score=linguistic_features['future_score'],
                action_density=linguistic_features['action_density'],
                certainty=linguistic_features['certainty'],
                enthusiasm=linguistic_features['enthusiasm'],
                engagement=behavioral_features['engagement'],
                thread_pos=behavioral_features['thread_pos'],
                upvote_ratio=behavioral_features['upvote_ratio'],
                optimism_score=optimism_score
            )
            
            results.append(features)
        
        return results
    
    def analyze_batch(self, threads: List[List[TextData]]) -> Dict[str, List[OptimismFeatures]]:
        results = {}
        
        for i, thread in enumerate(threads):
            thread_id = f"thread_{i}" if not thread else thread[0].thread_id
            results[thread_id] = self.analyze_thread(thread)
        
        return results
    
    def get_thread_summary(self, thread_features: List[OptimismFeatures]) -> Dict[str, float]:
        if not thread_features:
            return {}
        
        optimism_scores = [f.optimism_score for f in thread_features]
        
        return {
            'mean_optimism': np.mean(optimism_scores),
            'median_optimism': np.median(optimism_scores),
            'std_optimism': np.std(optimism_scores),
            'min_optimism': np.min(optimism_scores),
            'max_optimism': np.max(optimism_scores),
            'optimism_trend': self.calc_optimism_trend(optimism_scores)
        }
    
    def calc_optimism_trend(self, scores: List[float]) -> float:
        if len(scores) < 2:
            return 0
        
        x = np.arange(len(scores))
        correlation = np.corrcoef(x, scores)[0, 1]
        return correlation if not np.isnan(correlation) else 0

def create_sample_data() -> List[List[TextData]]:
    base_time = datetime.now()
    
    tesla_thread = [
        TextData(
            text="TSLA to the moon! Just read about their new 4680 battery breakthrough. This is absolutely game-changing tech that will revolutionize not just EVs but energy storage globally. I'm incredibly bullish on this - we're looking at potential 50% cost reduction and 5x energy density improvements. The future is electric and Tesla is leading the charge! Anyone else loading up on calls? This stock is going to $500 easy within 6 months. The competition can't even come close to matching this innovation.",
            timestamp=base_time - timedelta(hours=3),
            thread_id="tesla_discussion",
            user_id="tesla_bull_2024",
            upvotes=234,
            downvotes=45
        ),
        TextData(
            text="Honestly I'm skeptical about these battery claims. Tesla has overpromised before (remember the Cybertruck timeline?). The 4680 cells have been 'coming soon' for years now. I think the market is getting ahead of itself here. Competition from Ford, GM, and especially Chinese manufacturers is heating up. Tesla's valuation is already stretched thin at current levels. Might be time to take some profits...",
            timestamp=base_time - timedelta(hours=2, minutes=45),
            thread_id="tesla_discussion",
            user_id="realistic_trader",
            upvotes=89,
            downvotes=156
        ),
        TextData(
            text="I get your concerns but you're missing the bigger picture here! Tesla isn't just a car company - they're an energy and AI company. The FSD progress alone is worth the current valuation. Plus Elon just confirmed on Twitter that production scaling is ahead of schedule. I've been holding since $40 and every dip has been a buying opportunity. This company will definitely hit $1000+ in the next 2 years. The robotaxi network is going to print money!",
            timestamp=base_time - timedelta(hours=2, minutes=30),
            thread_id="tesla_discussion", 
            user_id="diamond_hands_elon",
            upvotes=178,
            downvotes=67
        )
    ]
    
    restaurant_thread = [
        TextData(
            text="OMG just went to the new ramen place downtown and I'm OBSESSED!! The tonkotsu broth is absolutely incredible - rich, creamy, and perfectly balanced. You can tell they're using authentic techniques and premium ingredients. The noodles have that perfect chewy texture and the chashu melts in your mouth. Honestly, this might be the best ramen I've had outside of Japan! The chef trained in Tokyo for 5 years and it shows. I'm definitely going back this weekend and bringing all my friends. This place is going to be HUGE! Already planning my next order",
            timestamp=base_time - timedelta(hours=4),
            thread_id="ramen_opening",
            user_id="foodie_explorer",
            upvotes=156,
            downvotes=8
        ),
        TextData(
            text="Eh, I went there yesterday and wasn't that impressed. The broth was decent but nothing special, and $18 for a bowl of ramen seems pretty steep. Service was slow too - waited 45 minutes for our food. Maybe it was just opening week issues but for that price I expected more. There are better ramen places across town for half the cost. Probably won't be going back anytime soon.",
            timestamp=base_time - timedelta(hours=3, minutes=30),
            thread_id="ramen_opening",
            user_id="budget_eater",
            upvotes=67,
            downvotes=89
        )
    ]
    
    phone_thread = [
        TextData(
            text="HOLY SHIT the new Samsung Galaxy Ultra is absolutely INSANE!! Just got mine and the camera is legitimately mind-blowing. The 200MP main sensor with AI processing is capturing details I didn't even know were possible on a phone. Shot some photos last night and people are asking if I used a professional camera! The 100x zoom is wild too - can literally read street signs from blocks away. Battery life is incredible, lasting 2+ days with heavy use. This is hands down the best phone I've ever owned. Samsung has completely destroyed the competition this year! iPhone who???",
            timestamp=base_time - timedelta(hours=5),
            thread_id="galaxy_launch",
            user_id="tech_reviewer_pro",
            upvotes=445,
            downvotes=67
        ),
        TextData(
            text="Idk man, seems like typical Samsung overhype to me. The specs look good on paper but real-world performance is what matters. Also that price tag is absolutely ridiculous - $1400+ for a phone? For that money I could buy a decent laptop. Plus Samsung's track record with software updates is still sketchy. My S20 barely gets security patches anymore. Think I'll stick with my iPhone 14 Pro for now, at least I know it'll work reliably for years.",
            timestamp=base_time - timedelta(hours=4, minutes=20),
            thread_id="galaxy_launch",
            user_id="apple_loyalist",
            upvotes=234,
            downvotes=178
        )
    ]
    
    coffee_thread = [
        TextData(
            text="The new coffee shop on 5th Street is absolutely amazing! Their single-origin Ethiopian beans are incredible - so much complexity and flavor. The baristas actually know what they're doing (finally!) and the latte art is Instagram-worthy. Plus the owner is super passionate about sustainability and fair trade. This is exactly what our neighborhood needed! Already planning to work from there tomorrow. Supporting local businesses like this gives me so much hope for our community!",
            timestamp=base_time - timedelta(hours=6),
            thread_id="local_coffee",
            user_id="coffee_addict_daily",
            upvotes=89,
            downvotes=12
        ),
        TextData(
            text="Tried it yesterday and honestly wasn't blown away. Coffee was okay but nothing special, and $6 for a latte is pretty steep for this area. The atmosphere is nice but seating is limited. Starbucks down the street is cheaper and more convenient. Don't really see what all the hype is about tbh.",
            timestamp=base_time - timedelta(hours=5, minutes=30),
            thread_id="local_coffee",
            user_id="practical_coffee_drinker",
            upvotes=34,
            downvotes=56
        )
    ]
    
    return [tesla_thread, restaurant_thread, phone_thread, coffee_thread]

def main():
    analyzer = OptimismAnalyzer()
    
    sample_threads = create_sample_data()
    thread_names = ["Tesla Stock Discussion", "New Ramen Restaurant", "Galaxy Phone Launch", "Local Coffee Shop"]
    
    output_lines = []
    output_lines.append("Optimism Analysis Report")
    output_lines.append("=" * 50)
    output_lines.append("")
    
    overall_results = {}
    
    for i, (thread, name) in enumerate(zip(sample_threads, thread_names)):
        output_lines.append(f"THREAD {i+1}: {name}")
        output_lines.append("-" * 40)
        
        results = analyzer.analyze_thread(thread)
        summary = analyzer.get_thread_summary(results)
        overall_results[name] = summary
        
        output_lines.append(f"Overall Thread Optimism: {summary['mean_optimism']:.3f}/1.0")
        trend_dir = "Growing" if summary['optimism_trend'] > 0.1 else "Declining" if summary['optimism_trend'] < -0.1 else "Stable"
        output_lines.append(f"Optimism Trend: {summary['optimism_trend']:.3f} ({trend_dir})")
        volatility = "High" if summary['std_optimism'] > 0.2 else "Low"
        output_lines.append(f"Volatility: {summary['std_optimism']:.3f} ({volatility})")
        output_lines.append(f"Peak Optimism: {summary['max_optimism']:.3f}")
        
        if summary['mean_optimism'] > 0.65:
            output_lines.append("INSIGHT: HIGH optimism - Strong positive sentiment")
        elif summary['mean_optimism'] > 0.45:
            output_lines.append("INSIGHT: MODERATE optimism - Mixed but generally positive")
        else:
            output_lines.append("INSIGHT: LOW optimism - Concerns or negativity present")
        
        sorted_results = sorted(zip(thread, results), key=lambda x: x[1].optimism_score, reverse=True)
        
        output_lines.append("")
        output_lines.append(f"Most Optimistic Post (Score: {sorted_results[0][1].optimism_score:.3f}):")
        output_lines.append(f"   '{sorted_results[0][0].text[:120]}...'")
        output_lines.append(f"   Upvotes: {sorted_results[0][0].upvotes} | Future Focus: {sorted_results[0][1].future_score:.2f} | Enthusiasm: {sorted_results[0][1].enthusiasm:.2f}")
        
        output_lines.append("")
        output_lines.append(f"Least Optimistic Post (Score: {sorted_results[-1][1].optimism_score:.3f}):")
        output_lines.append(f"   '{sorted_results[-1][0].text[:120]}...'")
        output_lines.append(f"   Upvotes: {sorted_results[-1][0].upvotes} | VADER: {sorted_results[-1][1].vader_comp:.2f}")
        
        output_lines.append("")
        output_lines.append("")
    
    output_lines.append("CROSS-THREAD ANALYSIS")
    output_lines.append("=" * 50)
    
    optimism_scores = [summary['mean_optimism'] for summary in overall_results.values()]
    trend_scores = [summary['optimism_trend'] for summary in overall_results.values()]
    
    output_lines.append(f"Average Optimism Across All Threads: {np.mean(optimism_scores):.3f}")
    output_lines.append(f"Average Trend Across All Threads: {np.mean(trend_scores):.3f}")
    
    ranked_threads = sorted(overall_results.items(), key=lambda x: x[1]['mean_optimism'], reverse=True)
    output_lines.append("")
    output_lines.append("THREAD RANKINGS BY OPTIMISM:")
    for i, (name, summary) in enumerate(ranked_threads):
        output_lines.append(f"   {i+1}. {name}: {summary['mean_optimism']:.3f}")
    
    output_lines.append("")
    output_lines.append("MARKET RESEARCH INSIGHTS:")
    if optimism_scores[0] > 0.6:
        output_lines.append("   Tesla: High optimism suggests strong investor confidence")
    if optimism_scores[1] > 0.6:
        output_lines.append("   Restaurant: High optimism indicates successful launch")
    if optimism_scores[2] > 0.6:
        output_lines.append("   Samsung: High optimism suggests positive market reception")
    if optimism_scores[3] > 0.6:
        output_lines.append("   Coffee Shop: High optimism indicates community support")
    
    output_lines.append("")
    output_lines.append(f"Analysis complete. Processed {sum(len(thread) for thread in sample_threads)} total posts across {len(sample_threads)} discussions.")
    
    with open("report.txt", "w") as f:
        f.write("\n".join(output_lines))
    
    print("Analysis completed. Report saved to report.txt")
    
    return overall_results

if __name__ == "__main__":
    main()