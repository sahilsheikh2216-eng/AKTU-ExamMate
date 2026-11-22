"""
AKTU ExamMate — Study Planner Service

Generates a personalised day-by-day study plan based on
question repetition frequency and available preparation time.
"""
import os
import json
import math
from typing import List, Dict, Optional
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPEATED_JSON, QUESTIONS_JSON


def load_repeated_questions() -> List[Dict]:
    """Load repeated question groups."""
    if not os.path.exists(REPEATED_JSON):
        return []
    with open(REPEATED_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_all_questions() -> List[Dict]:
    """Load all extracted questions."""
    if not os.path.exists(QUESTIONS_JSON):
        return []
    with open(QUESTIONS_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def categorize_questions(repeated_groups: List[Dict], 
                          all_questions: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Categorize questions into priority tiers based on repetition frequency.
    
    Tiers:
    - critical: Appeared in 4+ distinct years — highest exam frequency
    - high: Appeared in 3 distinct years
    - medium: Appeared in 2 distinct years
    - low: Appeared in only 1 year (unique questions)
    """
    tiers = {
        'critical': [],
        'high': [],
        'medium': [],
        'low': []
    }
    
    grouped_canonical_texts = set()
    grouped_variation_texts = set()
    
    for group in repeated_groups:
        distinct_years = group.get('distinct_year_count', 1)
        years = group.get('years_appeared', [])
        
        # Calculate recency score (more recent = higher priority)
        max_year = max(years) if years else 2020
        recent_years_count = sum(1 for y in years if y >= 2022)
        recency_bonus = (max_year - 2015) / 11.0 + (recent_years_count * 0.1)
        
        entry = {
            'group_id': group['group_id'],
            'question': group['canonical_question'],
            'distinct_years': distinct_years,
            'years': years,
            'recency_score': recency_bonus,
            'match_type': group.get('match_type', 'unknown'),
            'source_files': group.get('source_files', []),
            'question_count': group.get('question_count', len(group.get('variations', [])))
        }
        
        if distinct_years >= 4:
            tiers['critical'].append(entry)
        elif distinct_years >= 3:
            tiers['high'].append(entry)
        elif distinct_years >= 2:
            tiers['medium'].append(entry)
        
        grouped_canonical_texts.add(group['canonical_question'].strip().lower())
        for v in group.get('variations', []):
            grouped_variation_texts.add(v.get('question', '').strip().lower())
    
    # Add unique (non-repeated) questions from all_questions
    seen_unique = set()
    for q in all_questions:
        q_text = q.get('english_question', '').strip()
        q_norm = q_text.lower()
        
        if (q_norm not in grouped_canonical_texts and 
            q_norm not in grouped_variation_texts and 
            q_norm not in seen_unique and
            len(q_text) >= 20):
            
            seen_unique.add(q_norm)
            year = int(q.get('year', 2020))
            tiers['low'].append({
                'group_id': None,
                'question': q_text,
                'distinct_years': 1,
                'years': [year],
                'recency_score': (year - 2015) / 11.0,
                'match_type': 'unique',
                'source_files': [q.get('source_file', '')],
                'question_count': 1
            })
    
    # Sort each tier: primary by recency & distinct years
    for tier_name, q_list in tiers.items():
        q_list.sort(key=lambda x: (x['distinct_years'], x['recency_score']), reverse=True)
    
    return tiers


def generate_study_plan(days_available: int, hours_per_day: float,
                         repeated_groups: List[Dict],
                         all_questions: List[Dict]) -> Dict:
    """
    Generate a personalised day-by-day study plan.
    
    Prioritisation:
    - Frequently repeated questions receive higher priority.
    - Questions repeated across multiple recent years receive attention.
    - Less frequent questions receive coverage as available study hours permit.
    - Includes revision and practice sessions.
    - Spreads high-priority questions throughout the preparation period.
    - Does not claim that repeated questions are guaranteed to appear again.
    """
    if days_available <= 0 or hours_per_day <= 0:
        return {"error": "Please provide valid days and hours (days >= 1, hours > 0)"}
    
    tiers = categorize_questions(repeated_groups, all_questions)
    
    total_hours = days_available * hours_per_day
    
    # Allocate revision days:
    # 1-3 days available: 0 dedicated revision days (embedded daily revision)
    # 4-7 days: 1 revision day
    # 8-15 days: 2 revision days
    # 16-30 days: 3-4 revision days
    # >30 days: ~15% revision days
    if days_available <= 3:
        revision_days = 0
    elif days_available <= 7:
        revision_days = 1
    elif days_available <= 14:
        revision_days = 2
    elif days_available <= 25:
        revision_days = 3
    else:
        revision_days = max(4, int(days_available * 0.15))
    
    study_days = days_available - revision_days
    if study_days < 1:
        study_days = 1
        revision_days = max(0, days_available - 1)
    
    # Realistic question studying pace:
    # Each question involves: reading, understanding concept, deriving/tracing/coding, note-taking.
    # ~25-30 mins per question on average.
    # Questions per study day: ~2 questions per study hour.
    # For example: 4 hours/day -> 8-10 questions per day.
    target_questions_per_day = max(3, int(hours_per_day * 2.2))
    total_capacity = target_questions_per_day * study_days
    
    # Select questions based on capacity and priority:
    # 1. Critical questions (appeared 4+ years): include all (or up to capacity)
    # 2. High questions (appeared 3 years): include all (or remaining capacity)
    # 3. Medium questions (appeared 2 years): include as many as fit
    # 4. Low questions (appeared 1 year): include most recent/important if capacity remains
    
    selected_critical = list(tiers['critical'])
    selected_high = list(tiers['high'])
    selected_medium = []
    selected_low = []
    
    current_count = len(selected_critical) + len(selected_high)
    
    if current_count > total_capacity:
        # Very short preparation time: prioritize critical, then top high
        if len(selected_critical) >= total_capacity:
            selected_critical = selected_critical[:total_capacity]
            selected_high = []
        else:
            remaining = total_capacity - len(selected_critical)
            selected_high = selected_high[:remaining]
    else:
        remaining_capacity = total_capacity - current_count
        if remaining_capacity > 0:
            selected_medium = tiers['medium'][:remaining_capacity]
            remaining_capacity -= len(selected_medium)
        
        if remaining_capacity > 0:
            # Sort low questions by recency
            sorted_low = sorted(tiers['low'], key=lambda x: x['recency_score'], reverse=True)
            selected_low = sorted_low[:remaining_capacity]
    
    # Summary of questions included in the plan
    tier_summary = {
        'critical': {
            'count': len(selected_critical),
            'total_available': len(tiers['critical']),
            'description': 'Questions repeated across 4+ examination years',
            'estimated_hours': round(len(selected_critical) * 0.5, 1)
        },
        'high': {
            'count': len(selected_high),
            'total_available': len(tiers['high']),
            'description': 'Questions repeated across 3 examination years',
            'estimated_hours': round(len(selected_high) * 0.45, 1)
        },
        'medium': {
            'count': len(selected_medium),
            'total_available': len(tiers['medium']),
            'description': 'Questions repeated across 2 examination years',
            'estimated_hours': round(len(selected_medium) * 0.4, 1)
        },
        'low': {
            'count': len(selected_low),
            'total_available': len(tiers['low']),
            'description': 'Single-appearance questions included for topic coverage',
            'estimated_hours': round(len(selected_low) * 0.35, 1)
        }
    }
    
    # Distribute questions across study days:
    # RULE: Spread high-priority questions throughout the preparation period!
    # We round-robin distribute critical and high questions across ALL study days.
    day_assignments = [[] for _ in range(study_days)]
    
    # 1. Distribute critical questions round-robin
    for i, q in enumerate(selected_critical):
        day_idx = i % study_days
        day_assignments[day_idx].append({'tier': 'critical', **q})
    
    # 2. Distribute high questions round-robin (offset to balance)
    offset = (len(selected_critical) % study_days)
    for i, q in enumerate(selected_high):
        day_idx = (i + offset) % study_days
        day_assignments[day_idx].append({'tier': 'high', **q})
    
    # 3. Distribute medium questions
    offset_m = (offset + len(selected_high)) % study_days
    for i, q in enumerate(selected_medium):
        day_idx = (i + offset_m) % study_days
        day_assignments[day_idx].append({'tier': 'medium', **q})
    
    # 4. Distribute low questions
    offset_l = (offset_m + len(selected_medium)) % study_days
    for i, q in enumerate(selected_low):
        day_idx = (i + offset_l) % study_days
        day_assignments[day_idx].append({'tier': 'low', **q})
    
    # Build daily schedule
    daily_plan = []
    
    for day_num in range(1, study_days + 1):
        day_idx = day_num - 1
        tasks_raw = day_assignments[day_idx]
        
        # Sort day's tasks: critical first, then high, medium, low
        tier_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        tasks_raw.sort(key=lambda t: tier_order.get(t['tier'], 9))
        
        total_day_minutes = int(hours_per_day * 60)
        
        # Reserve ~20% of daily time (or 30 mins) for daily practice & self-test
        practice_minutes = max(20, min(45, int(total_day_minutes * 0.18)))
        study_minutes_pool = max(30, total_day_minutes - practice_minutes)
        
        formatted_tasks = []
        if tasks_raw:
            mins_per_question = max(15, study_minutes_pool // len(tasks_raw))
            for task in tasks_raw:
                years_str = ', '.join(str(y) for y in task.get('years', []))
                distinct_y = task.get('distinct_years', 1)
                
                if distinct_y >= 4:
                    freq_note = f"⭐ Appeared in {distinct_y} years ({years_str})"
                elif distinct_y >= 3:
                    freq_note = f"🔥 Appeared in {distinct_y} years ({years_str})"
                elif distinct_y >= 2:
                    freq_note = f"📌 Appeared in {distinct_y} years ({years_str})"
                else:
                    freq_note = f"📝 Appeared in {years_str}"
                
                formatted_tasks.append({
                    'activity': task['question'],
                    'tier': task['tier'],
                    'distinct_years': distinct_y,
                    'years_appeared': task.get('years', []),
                    'estimated_minutes': mins_per_question,
                    'frequency_note': freq_note,
                    'source_files': task.get('source_files', [])
                })
        
        # Add the daily practice session
        formatted_tasks.append({
            'activity': "Practice & Self-Test: Solve today's algorithms/diagrams on paper without looking at solutions",
            'tier': 'practice',
            'distinct_years': 0,
            'years_appeared': [],
            'estimated_minutes': practice_minutes,
            'frequency_note': "Daily Consolidation",
            'source_files': []
        })
        
        # Day focus title
        crit_count = sum(1 for t in tasks_raw if t['tier'] == 'critical')
        high_count = sum(1 for t in tasks_raw if t['tier'] == 'high')
        if crit_count > 0 and high_count > 0:
            day_focus = "⭐ Core High-Frequency Questions & Concepts"
        elif crit_count > 0:
            day_focus = "⭐ Critical Repeated Questions"
        elif high_count > 0:
            day_focus = "🔥 High-Priority Repetition Questions"
        else:
            day_focus = "📌 Moderate-Frequency & Coverage Topics"
        
        daily_plan.append({
            'day': day_num,
            'type': 'study',
            'hours': hours_per_day,
            'focus': day_focus,
            'question_count': len(tasks_raw),
            'tasks': formatted_tasks,
            'summary': f"{len(tasks_raw)} questions scheduled + {practice_minutes}m self-test"
        })
    
    # Add dedicated revision days
    for rev_day in range(1, revision_days + 1):
        day_num = study_days + rev_day
        total_rev_minutes = int(hours_per_day * 60)
        
        if rev_day == revision_days:
            # Final day
            focus = "🏁 Final Pre-Exam Review & Mock Paper"
            tasks = [
                {
                    'activity': "Full Mock Test: Pick 1 complete previous year paper and solve under 3-hour exam conditions",
                    'tier': 'revision',
                    'distinct_years': 0,
                    'years_appeared': [],
                    'estimated_minutes': min(180, total_rev_minutes - 40),
                    'frequency_note': "Simulated Exam",
                    'source_files': []
                },
                {
                    'activity': "Review formula sheet, tree traversal algorithms, and asymptotic complexity table",
                    'tier': 'revision',
                    'distinct_years': 0,
                    'years_appeared': [],
                    'estimated_minutes': 40,
                    'frequency_note': "Quick Formula Review",
                    'source_files': []
                }
            ]
        elif rev_day == 1 and revision_days > 1:
            focus = "📖 Comprehensive Revision: Top Repeated Questions"
            tasks = [
                {
                    'activity': "Re-derive and trace all ⭐ Critical repeated algorithms (Heap Sort, Quick Sort, Infix-to-Postfix, AVL)",
                    'tier': 'revision',
                    'distinct_years': 0,
                    'years_appeared': [],
                    'estimated_minutes': total_rev_minutes // 2,
                    'frequency_note': "Top Algorithms",
                    'source_files': []
                },
                {
                    'activity': "Review Queue & Linked List implementations and edge cases (overflow, underflow, circular)",
                    'tier': 'revision',
                    'distinct_years': 0,
                    'years_appeared': [],
                    'estimated_minutes': total_rev_minutes // 2,
                    'frequency_note': "Data Structures Core",
                    'source_files': []
                }
            ]
        else:
            focus = f"🔄 Targeted Review & Practice Session {rev_day}"
            tasks = [
                {
                    'activity': "Practice tree traversals (construct tree from Inorder & Preorder) and graph algorithms (BFS, DFS, Dijkstra)",
                    'tier': 'revision',
                    'distinct_years': 0,
                    'years_appeared': [],
                    'estimated_minutes': total_rev_minutes // 2,
                    'frequency_note': "Trees & Graphs Practice",
                    'source_files': []
                },
                {
                    'activity': "Review weak questions and re-solve problems flagged during earlier study days",
                    'tier': 'revision',
                    'distinct_years': 0,
                    'years_appeared': [],
                    'estimated_minutes': total_rev_minutes // 2,
                    'frequency_note': "Targeted Correction",
                    'source_files': []
                }
            ]
        
        daily_plan.append({
            'day': day_num,
            'type': 'revision',
            'hours': hours_per_day,
            'focus': focus,
            'question_count': 0,
            'tasks': tasks,
            'summary': f"Dedicated revision and practice session ({hours_per_day}h)"
        })
    
    total_questions_scheduled = sum(t['count'] for t in tier_summary.values())
    
    plan = {
        'total_days': days_available,
        'hours_per_day': hours_per_day,
        'total_study_hours': round(total_hours, 1),
        'study_days': study_days,
        'revision_days': revision_days,
        'total_questions_scheduled': total_questions_scheduled,
        'tier_summary': tier_summary,
        'daily_plan': daily_plan,
        'disclaimer': (
            "Important: This study plan is formulated based on historical question repetition patterns "
            "across 58 AKTU Data Structures previous examination papers (2015-2026). "
            "Past frequency does NOT guarantee that any specific question will appear in future examinations. "
            "Use this plan as an evidence-based preparation prioritization guide."
        )
    }
    
    return plan


if __name__ == "__main__":
    repeated = load_repeated_questions()
    all_qs = load_all_questions()
    
    print("Testing study planner for multiple durations...")
    for days, hours in [(3, 4), (7, 4), (14, 3.5), (30, 4)]:
        p = generate_study_plan(days, hours, repeated, all_qs)
        print(f"\n{'='*50}")
        print(f"Plan: {days} days, {hours}h/day (Total {p['total_study_hours']}h)")
        print(f"Study days: {p['study_days']}, Revision days: {p['revision_days']}")
        print(f"Total questions scheduled: {p['total_questions_scheduled']}")
        for t, d in p['tier_summary'].items():
            print(f"  {t.title()}: {d['count']} / {d['total_available']} ({d['description']})")
        print(f"Day 1: {p['daily_plan'][0]['focus']} — {len(p['daily_plan'][0]['tasks'])} tasks")
