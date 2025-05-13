import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
import io
import re

st.set_page_config(layout="wide", page_title="Теннисная аналитика")

# Получаем настройки из боковой панели
def add_settings_sidebar():
    st.sidebar.title("Настройки анализа")
    
    # Общие настройки
    st.sidebar.header("Общие настройки")
    
    show_help = st.sidebar.checkbox("Показывать подсказки", value=True)
    
    # Настройки визуализации
    st.sidebar.header("Визуализация")
    
    color_scheme = st.sidebar.selectbox(
        "Цветовая схема графиков",
        ["Стандартная", "Синий-Красный", "Зеленый-Оранжевый", "Пастельная"]
    )
    
    chart_height = st.sidebar.slider("Высота графиков", 300, 800, 400, 50)
    
    # Настройки рекомендаций
    st.sidebar.header("Рекомендации")
    
    recommendation_detail = st.sidebar.select_slider(
        "Детализация рекомендаций",
        options=["Минимальная", "Средняя", "Подробная"],
        value="Средняя"
    )
    
    # Возвращаем настройки в виде словаря
    return {
        "show_help": show_help,
        "color_scheme": color_scheme,
        "chart_height": chart_height,
        "recommendation_detail": recommendation_detail
    }

# Получаем цветовую схему на основе настроек
def get_color_scheme(settings):
    if settings["color_scheme"] == "Синий-Красный":
        return {"player1": "#1e40af", "player2": "#be123c"}
    elif settings["color_scheme"] == "Зеленый-Оранжевый":
        return {"player1": "#15803d", "player2": "#c2410c"}
    elif settings["color_scheme"] == "Пастельная":
        return {"player1": "#0ea5e9", "player2": "#f472b6"}
    else:  # Стандартная
        return {"player1": "#0088FE", "player2": "#FF8042"}

# Пороговые значения для разных показателей
thresholds = {
    'first_serve_pct': {'low': 50, 'medium': 60, 'high': 70},
    'second_serve_pct': {'low': 80, 'medium': 85, 'high': 90},
    'first_serve_won_pct': {'low': 60, 'medium': 70, 'high': 75},
    'second_serve_won_pct': {'low': 40, 'medium': 50, 'high': 60},
    'break_point_conversion': {'low': 30, 'medium': 40, 'high': 50},
    'forehand_winners_ratio': {'low': 0.5, 'medium': 1.0, 'high': 1.5},
    'long_rally_win_pct': {'low': 40, 'medium': 50, 'high': 60},
}

def analyze_match_data(df):
    """
    Анализирует данные матча из CSV и возвращает статистику для обоих игроков.
    """
    # Получаем имена игроков
    players = list(df['Player_1'].unique())
    
    # Инициализируем словарь для статистики
    player_stats = {player: {} for player in players}
    
    # Группируем розыгрыши
    points = []
    current_point = None
    
    for _, row in df.iterrows():
        # Начало нового розыгрыша
        if isinstance(row['Serve'], str) and row['Serve'] in ['1st', '2nd', '1st Serve', '2nd Serve']:
            if current_point:
                points.append(current_point)
            current_point = {'server': row['Player_1'], 'actions': [row.to_dict()]}
        elif current_point:
            current_point['actions'].append(row.to_dict())
    
    # Добавляем последний розыгрыш
    if current_point:
        points.append(current_point)
    
    # Анализ подачи
    for player in players:
        player_stats[player]['first_serve_total'] = 0
        player_stats[player]['first_serve_in'] = 0
        player_stats[player]['first_serve_won'] = 0
        player_stats[player]['second_serve_total'] = 0
        player_stats[player]['second_serve_in'] = 0
        player_stats[player]['second_serve_won'] = 0
        player_stats[player]['aces'] = 0
        player_stats[player]['double_faults'] = 0
        player_stats[player]['serve_zones'] = {}
        player_stats[player]['shot_types'] = {}
        player_stats[player]['shot_combinations'] = {}
        player_stats[player]['points_by_rally_length'] = {'1-3': 0, '4-6': 0, '7-9': 0, '10+': 0}
        player_stats[player]['wins_by_rally_length'] = {'1-3': 0, '4-6': 0, '7-9': 0, '10+': 0}
        # Для анализа ключевых моментов
        player_stats[player]['break_points'] = {'faced': 0, 'saved': 0, 'converted': 0}
        player_stats[player]['game_points'] = {'faced': 0, 'saved': 0, 'converted': 0}
        player_stats[player]['key_shots'] = {}
        player_stats[player]['pressure_points_won'] = 0
        player_stats[player]['pressure_points_total'] = 0
    
    # Анализируем каждый розыгрыш
    for point in points:
        server = point['server']
        returner = [p for p in players if p != server][0] if len(players) > 1 else None
        
        # Анализ подачи
        first_serve = next((a for a in point['actions'] if isinstance(a.get('Serve'), str) and a['Serve'] in ['1st', '1st Serve']), None)
        second_serve = next((a for a in point['actions'] if isinstance(a.get('Serve'), str) and a['Serve'] in ['2nd', '2nd Serve']), None)
        
        if first_serve is not None:
            player_stats[server]['first_serve_total'] += 1
            
            # Анализ зоны подачи
            serve_zone = first_serve.get('Serve Zone')
            if isinstance(serve_zone, str) and serve_zone != '-':
                if serve_zone not in player_stats[server]['serve_zones']:
                    player_stats[server]['serve_zones'][serve_zone] = 0
                player_stats[server]['serve_zones'][serve_zone] += 1
            
            serve_result = first_serve.get('Serve Result')
            if isinstance(serve_result, str) and serve_result in ['In', 'In Play']:
                player_stats[server]['first_serve_in'] += 1
            elif isinstance(serve_result, str) and serve_result == 'Ace':
                player_stats[server]['first_serve_in'] += 1
                player_stats[server]['aces'] += 1
                player_stats[server]['first_serve_won'] += 1
        
        if second_serve is not None:
            player_stats[server]['second_serve_total'] += 1
            
            serve_result = second_serve.get('Serve Result')
            if isinstance(serve_result, str) and serve_result in ['In', 'In Play']:
                player_stats[server]['second_serve_in'] += 1
            elif isinstance(serve_result, str) and serve_result == 'Double Fault':
                player_stats[server]['double_faults'] += 1
        
        # Определение победителя розыгрыша
        winner = None
        last_action = point['actions'][-1]
        
        if isinstance(last_action.get('Finish Type'), str) and last_action['Finish Type'] == 'Winner':
            winner = last_action['Player_1']
        elif isinstance(last_action.get('Finish Type'), str) and last_action['Finish Type'] in ['Forced Error', 'Unforced Error']:
            winner = [p for p in players if p != last_action['Player_1']][0] if len(players) > 1 else None
        
        # Анализ счета в гейме для определения ключевых моментов
        game_score = None
        for action in point['actions']:
            if isinstance(action.get('Game Score'), str) and action['Game Score'] != '-':
                game_score = action['Game Score']
                break
        
        if game_score and returner:
            # Проверяем, является ли это брейк-пойнтом
            is_break_point = False
            is_game_point = False
            
            # Попробуем определить брейк-пойнты и гейм-пойнты по стандартным обозначениям
            try:
                if '40-A' in game_score or 'A-40' in game_score or '30-40' in game_score or '40-30' in game_score or '15-40' in game_score or '40-15' in game_score or '0-40' in game_score or '40-0' in game_score:
                    if '40-A' in game_score or '30-40' in game_score or '15-40' in game_score or '0-40' in game_score:
                        is_break_point = True
                        player_stats[server]['break_points']['faced'] += 1
                        player_stats[returner]['break_points']['faced'] += 1
                    
                    if 'A-40' in game_score or '40-30' in game_score or '40-15' in game_score or '40-0' in game_score:
                        is_game_point = True
                        player_stats[server]['game_points']['faced'] += 1
            except:
                pass
            
            # Обновляем статистику по ключевым моментам
            if winner and (is_break_point or is_game_point):
                if is_break_point:
                    if winner == returner:
                        player_stats[returner]['break_points']['converted'] += 1
                    else:
                        player_stats[server]['break_points']['saved'] += 1
                
                if is_game_point:
                    if winner == server:
                        player_stats[server]['game_points']['converted'] += 1
                    else:
                        player_stats[returner]['game_points']['saved'] += 1
        
        # Обновление статистики по победителю розыгрыша
        if winner:
            # Определяем длину розыгрыша
            shot_count = sum(1 for a in point['actions'] if isinstance(a.get('Shot Type'), str) and a['Shot Type'] != '-')
            
            if shot_count <= 3:
                rally_length = '1-3'
            elif shot_count <= 6:
                rally_length = '4-6'
            elif shot_count <= 9:
                rally_length = '7-9'
            else:
                rally_length = '10+'
            
            # Обновляем статистику по длине розыгрыша
            for player in players:
                player_stats[player]['points_by_rally_length'][rally_length] += 1
            
            # Отмечаем победителя
            player_stats[winner]['wins_by_rally_length'][rally_length] += 1
            
            # Обновляем статистику подачи
            if winner == server:
                if first_serve and isinstance(first_serve.get('Serve Result'), str) and first_serve['Serve Result'] in ['In', 'In Play']:
                    player_stats[server]['first_serve_won'] += 1
                elif second_serve and isinstance(second_serve.get('Serve Result'), str) and second_serve['Serve Result'] in ['In', 'In Play']:
                    player_stats[server]['second_serve_won'] += 1
        
        # Анализ типов ударов
        for action in point['actions']:
            shot_type = action.get('Shot Type')
            if isinstance(shot_type, str) and shot_type != '-':
                player = action['Player_1']
                
                if shot_type not in player_stats[player]['shot_types']:
                    player_stats[player]['shot_types'][shot_type] = 0
                player_stats[player]['shot_types'][shot_type] += 1
                
                # Анализ ключевых ударов
                if game_score and (is_break_point or is_game_point):
                    if shot_type not in player_stats[player]['key_shots']:
                        player_stats[player]['key_shots'][shot_type] = {'total': 0, 'won': 0}
                    
                    player_stats[player]['key_shots'][shot_type]['total'] += 1
                    
                    if winner == player:
                        player_stats[player]['key_shots'][shot_type]['won'] += 1
                
                # Обновляем статистику по напряженным моментам
                if game_score and (is_break_point or is_game_point or '30-30' in game_score or '40-40' in game_score):
                    player_stats[player]['pressure_points_total'] += 1
                    if winner == player:
                        player_stats[player]['pressure_points_won'] += 1
        
        # Анализ комбинаций ударов
        for i in range(len(point['actions']) - 1):
            curr_shot = point['actions'][i].get('Shot Type')
            next_shot = point['actions'][i+1].get('Shot Type')
            
            if (isinstance(curr_shot, str) and curr_shot != '-' and 
                isinstance(next_shot, str) and next_shot != '-'):
                
                player = point['actions'][i]['Player_1']
                combo = f"{curr_shot} → {next_shot}"
                
                if combo not in player_stats[player]['shot_combinations']:
                    player_stats[player]['shot_combinations'][combo] = {'count': 0, 'wins': 0}
                
                player_stats[player]['shot_combinations'][combo]['count'] += 1
                
                if winner == player:
                    player_stats[player]['shot_combinations'][combo]['wins'] += 1
    
    # Рассчитываем проценты и соотношения
    for player in players:
        # Процент подач
        if player_stats[player]['first_serve_total'] > 0:
            player_stats[player]['first_serve_pct'] = round(
                player_stats[player]['first_serve_in'] / player_stats[player]['first_serve_total'] * 100, 1
            )
        else:
            player_stats[player]['first_serve_pct'] = 0
            
        if player_stats[player]['second_serve_total'] > 0:
            player_stats[player]['second_serve_pct'] = round(
                player_stats[player]['second_serve_in'] / player_stats[player]['second_serve_total'] * 100, 1
            )
        else:
            player_stats[player]['second_serve_pct'] = 0
            
        # Процент выигранных очков на подаче
        if player_stats[player]['first_serve_in'] > 0:
            player_stats[player]['first_serve_won_pct'] = round(
                player_stats[player]['first_serve_won'] / player_stats[player]['first_serve_in'] * 100, 1
            )
        else:
            player_stats[player]['first_serve_won_pct'] = 0
            
        if player_stats[player]['second_serve_in'] > 0:
            player_stats[player]['second_serve_won_pct'] = round(
                player_stats[player]['second_serve_won'] / player_stats[player]['second_serve_in'] * 100, 1
            )
        else:
            player_stats[player]['second_serve_won_pct'] = 0
        
        # Расчет процента выигранных розыгрышей по длине
        for length in player_stats[player]['points_by_rally_length']:
            if player_stats[player]['points_by_rally_length'][length] > 0:
                player_stats[player][f'{length}_rally_win_pct'] = round(
                    player_stats[player]['wins_by_rally_length'][length] / 
                    player_stats[player]['points_by_rally_length'][length] * 100, 1
                )
            else:
                player_stats[player][f'{length}_rally_win_pct'] = 0
                
        # Обобщенный показатель для длинных розыгрышей (4+ ударов)
        long_rally_wins = sum(player_stats[player]['wins_by_rally_length'][l] 
                            for l in ['4-6', '7-9', '10+'])
        long_rally_points = sum(player_stats[player]['points_by_rally_length'][l] 
                              for l in ['4-6', '7-9', '10+'])
        
        if long_rally_points > 0:
            player_stats[player]['long_rally_win_pct'] = round(
                long_rally_wins / long_rally_points * 100, 1
            )
        else:
            player_stats[player]['long_rally_win_pct'] = 0
        
        # Расчет выигрышей комбинаций
        for combo in player_stats[player]['shot_combinations']:
            combo_stats = player_stats[player]['shot_combinations'][combo]
            if combo_stats['count'] > 0:
                combo_stats['win_percentage'] = round(
                    combo_stats['wins'] / combo_stats['count'] * 100, 1
                )
            else:
                combo_stats['win_percentage'] = 0
                
        # Расчет статистики по ключевым ударам
        for shot_type in player_stats[player]['key_shots']:
            shot_stats = player_stats[player]['key_shots'][shot_type]
            if shot_stats['total'] > 0:
                shot_stats['win_percentage'] = round(
                    shot_stats['won'] / shot_stats['total'] * 100, 1
                )
            else:
                shot_stats['win_percentage'] = 0
                
        # Процент выигранных очков под давлением
        if player_stats[player]['pressure_points_total'] > 0:
            player_stats[player]['pressure_points_pct'] = round(
                player_stats[player]['pressure_points_won'] / player_stats[player]['pressure_points_total'] * 100, 1
            )
        else:
            player_stats[player]['pressure_points_pct'] = 0
    
    return player_stats

def generate_player_recommendations(player_stats, opponent_stats=None, detail_level="Средняя"):
    """
    Генерирует рекомендации для игрока на основе его статистики
    и опционально статистики соперника.
    
    Args:
        player_stats: Статистика игрока
        opponent_stats: Статистика соперника
        detail_level: Уровень детализации рекомендаций ("Минимальная", "Средняя", "Подробная")
    """
    recommendations = {
        'strengths': [],        # Сильные стороны
        'improvements': [],     # Области для улучшения
        'tactics': [],          # Тактические рекомендации
        'training_focus': [],   # Фокус тренировок
        'mental_game': []       # Ментальный аспект
    }
    
    # Анализ подачи
    if player_stats['first_serve_pct'] < thresholds['first_serve_pct']['low']:
        recommendations['improvements'].append(
            f"Улучшить процент первой подачи (текущий: {player_stats['first_serve_pct']}%). "
            f"Сосредоточиться на технике и стабильности."
        )
        recommendations['training_focus'].append("Работа над первой подачей")
    elif player_stats['first_serve_pct'] > thresholds['first_serve_pct']['high']:
        recommendations['strengths'].append(
            f"Высокий процент первой подачи ({player_stats['first_serve_pct']}%). "
            f"Продолжать использовать это как преимущество."
        )
    
    # Анализ второй подачи
    if player_stats['second_serve_won_pct'] < thresholds['second_serve_won_pct']['low']:
        recommendations['improvements'].append(
            f"Низкий процент выигранных очков на второй подаче ({player_stats['second_serve_won_pct']}%). "
            f"Улучшить качество и вариативность второй подачи."
        )
        recommendations['training_focus'].append("Работа над второй подачей")
    
    # Анализ зон подачи
    serve_zones = player_stats.get('serve_zones', {})
    if serve_zones:
        total_serves = sum(serve_zones.values())
        if total_serves > 0:
            zone_percentages = {zone: count/total_serves*100 for zone, count in serve_zones.items()}
            max_zone_pct = max(zone_percentages.values())
            if max_zone_pct > 60:
                max_zone = max(zone_percentages, key=zone_percentages.get)
                recommendations['improvements'].append(
                    f"Чрезмерная концентрация подач в зону {max_zone} ({max_zone_pct:.1f}%). "
                    f"Увеличить вариативность подачи."
                )
    
    # Сравнение с соперником (если доступно)
    if opponent_stats:
        # Сравнение эффективности форхенда
        player_forehand = player_stats.get('shot_types', {}).get('Forehand', 0)
        opponent_forehand = opponent_stats.get('shot_types', {}).get('Forehand', 0)
        
        if player_forehand > opponent_forehand * 1.5 and player_forehand > 5:
            recommendations['strengths'].append(
                "Значительное преимущество в эффективности форхенда. "
                "Использовать форхенд как основное оружие."
            )
            recommendations['tactics'].append(
                "Строить розыгрыши через форхенд, искать возможности для атаки с форхенда"
            )
        
        # Сравнение эффективности в длинных розыгрышах
        if (player_stats.get('long_rally_win_pct', 0) > 
            opponent_stats.get('long_rally_win_pct', 0) + 20):
            recommendations['tactics'].append(
                f"Значительное преимущество в длинных розыгрышах " 
                f"({player_stats.get('long_rally_win_pct', 0)}% vs "
                f"{opponent_stats.get('long_rally_win_pct', 0)}%). "
                f"Стремиться к затяжным обменам ударами."
            )
        elif (player_stats.get('long_rally_win_pct', 0) < 
              opponent_stats.get('long_rally_win_pct', 0) - 20):
            recommendations['tactics'].append(
                f"Слабая эффективность в длинных розыгрышах "
                f"({player_stats.get('long_rally_win_pct', 0)}% vs "
                f"{opponent_stats.get('long_rally_win_pct', 0)}%). "
                f"Избегать затяжных обменов, играть более агрессивно."
            )
            recommendations['training_focus'].append(
                "Физическая подготовка и выносливость для длинных розыгрышей"
            )
        
        # Анализ брейк-пойнтов
        if player_stats['break_points']['faced'] > 2:
            bp_conv_pct = player_stats['break_points']['converted'] / player_stats['break_points']['faced'] * 100 if player_stats['break_points']['faced'] > 0 else 0
            if bp_conv_pct < 30:
                recommendations['mental_game'].append(
                    f"Низкий процент реализации брейк-пойнтов ({bp_conv_pct:.1f}%). "
                    f"Работать над концентрацией в ключевые моменты."
                )
            elif bp_conv_pct > 60:
                recommendations['strengths'].append(
                    f"Высокий процент реализации брейк-пойнтов ({bp_conv_pct:.1f}%). "
                    f"Хорошая психологическая устойчивость в ключевые моменты."
                )
    
    # Анализ типов ударов
    shot_types = player_stats.get('shot_types', {})
    if shot_types:
        total_shots = sum(shot_types.values())
        if total_shots > 0:
            for shot_type, count in shot_types.items():
                shot_pct = count / total_shots * 100
                
                # Анализ распределения между форхендом и бэкхендом
                if shot_type == 'Forehand' and shot_pct > 65:
                    recommendations['strengths'].append(
                        f"Высокое использование форхенда ({shot_pct:.1f}% всех ударов). "
                        f"Продолжать строить игру через форхенд."
                    )
                elif shot_type == 'Backhand' and shot_pct > 65:
                    recommendations['strengths'].append(
                        f"Высокое использование бэкхенда ({shot_pct:.1f}% всех ударов). "
                        f"Продолжать строить игру через бэкхенд."
                    )
                
                # Проверка редко используемых типов ударов
                if shot_type in ['Slice', 'Drop Shot', 'Volley'] and shot_pct < 5:
                    recommendations['improvements'].append(
                        f"Редкое использование удара {shot_type} ({shot_pct:.1f}%). "
                        f"Добавить больше вариативности в игру."
                    )
                    recommendations['training_focus'].append(f"Развитие удара {shot_type}")
    
    # Анализ комбинаций ударов
    shot_combinations = player_stats.get('shot_combinations', {})
    if shot_combinations:
        # Найти наиболее успешные комбинации
        successful_combos = [(combo, stats) for combo, stats in shot_combinations.items() 
                            if stats.get('win_percentage', 0) > 60 and stats.get('count', 0) >= 3]
        
        if successful_combos:
            # Сортировка по проценту побед
            successful_combos.sort(key=lambda x: x[1].get('win_percentage', 0), reverse=True)
            top_combo, top_stats = successful_combos[0]
            
            recommendations['tactics'].append(
                f"Комбинация '{top_combo}' особенно эффективна "
                f"({top_stats.get('win_percentage', 0)}% успешности). "
                f"Использовать чаще в ключевые моменты."
            )
    
    # Анализ ключевых ударов
    key_shots = player_stats.get('key_shots', {})
    if key_shots and detail_level == "Подробная":
        best_key_shots = [(shot, stats) for shot, stats in key_shots.items() 
                         if stats.get('win_percentage', 0) > 60 and stats.get('total', 0) >= 2]
        
        worst_key_shots = [(shot, stats) for shot, stats in key_shots.items() 
                          if stats.get('win_percentage', 0) < 40 and stats.get('total', 0) >= 2]
        
        if best_key_shots:
            best_key_shots.sort(key=lambda x: x[1].get('win_percentage', 0), reverse=True)
            shot, stats = best_key_shots[0]
            recommendations['strengths'].append(
                f"Эффективное использование {shot} в ключевые моменты "
                f"({stats.get('win_percentage', 0)}% успешности)."
            )
        
        if worst_key_shots:
            worst_key_shots.sort(key=lambda x: x[1].get('win_percentage', 0))
            shot, stats = worst_key_shots[0]
            recommendations['improvements'].append(
                f"Низкая эффективность {shot} в ключевые моменты "
                f"({stats.get('win_percentage', 0)}% успешности). "
                f"Работать над стабильностью этого удара под давлением."
            )
    
    # Рекомендации по ментальной игре на основе паттернов
    if player_stats.get('first_serve_pct', 0) > 65 and player_stats.get('second_serve_won_pct', 0) < 40:
        recommendations['mental_game'].append(
            "Высокий риск на второй подаче может привести к неуверенности. "
            "Работать над психологической стабильностью при второй подаче."
        )
    
# Анализ выигрышей под давлением
    if player_stats.get('pressure_points_total', 0) > 5:
        pressure_pct = player_stats.get('pressure_points_pct', 0)
        if pressure_pct < 40:
            recommendations['mental_game'].append(
                f"Низкий процент выигрыша очков под давлением ({pressure_pct:.1f}%). "
                f"Работать над ментальной устойчивостью в ключевые моменты."
            )
        elif pressure_pct > 60:
            recommendations['strengths'].append(
                f"Высокий процент выигрыша очков под давлением ({pressure_pct:.1f}%). "
                f"Хорошая психологическая устойчивость."
            )
    
    # Фильтрация рекомендаций в соответствии с уровнем детализации
    if detail_level == "Минимальная":
        # Для минимальной детализации оставляем только самые важные рекомендации
        for category in recommendations:
            recommendations[category] = recommendations[category][:1]
    elif detail_level == "Средняя":
        # Для средней детализации ограничиваем количество рекомендаций
        for category in recommendations:
            recommendations[category] = recommendations[category][:2]
    
    return recommendations

# Создание визуализаций
def create_serve_stats_chart(player_stats, colors, height=400):
    """
    Создает график статистики подачи.
    """
    players = list(player_stats.keys())
    
    # Создаем данные для графика
    data = []
    for player in players:
        data.append({
            'Игрок': player,
            'Показатель': 'Процент первой подачи',
            'Значение': player_stats[player].get('first_serve_pct', 0)
        })
        data.append({
            'Игрок': player,
            'Показатель': 'Выигрыш на первой подаче (%)',
            'Значение': player_stats[player].get('first_serve_won_pct', 0)
        })
        data.append({
            'Игрок': player,
            'Показатель': 'Выигрыш на второй подаче (%)',
            'Значение': player_stats[player].get('second_serve_won_pct', 0)
        })
    
    df = pd.DataFrame(data)
    
    # Создаем график
    fig = px.bar(
        df, 
        x='Показатель', 
        y='Значение', 
        color='Игрок',
        barmode='group',
        color_discrete_map={players[0]: colors['player1'], players[1]: colors['player2']} if len(players) > 1 else None,
        height=height
    )
    
    fig.update_layout(
        title='Статистика подачи',
        xaxis_title=None,
        yaxis_title='Процент (%)',
        legend_title='Игрок',
        yaxis_range=[0, 100]
    )
    
    return fig

def create_rally_stats_chart(player_stats, colors, height=400):
    """
    Создает график статистики розыгрышей по длине.
    """
    players = list(player_stats.keys())
    
    # Создаем данные для графика
    data = []
    for player in players:
        for rally_length in ['1-3', '4-6', '7-9', '10+']:
            data.append({
                'Игрок': player,
                'Длина розыгрыша': rally_length,
                'Процент выигрыша': player_stats[player].get(f'{rally_length}_rally_win_pct', 0)
            })
    
    df = pd.DataFrame(data)
    
    # Создаем график
    fig = px.line(
        df, 
        x='Длина розыгрыша', 
        y='Процент выигрыша', 
        color='Игрок',
        markers=True,
        color_discrete_map={players[0]: colors['player1'], players[1]: colors['player2']} if len(players) > 1 else None,
        height=height
    )
    
    fig.update_layout(
        title='Эффективность по длине розыгрыша',
        xaxis_title='Количество ударов в розыгрыше',
        yaxis_title='Процент выигранных розыгрышей (%)',
        legend_title='Игрок',
        yaxis_range=[0, 100]
    )
    
    return fig

def create_shot_types_chart(player_stats, colors, height=400):
    """
    Создает график распределения типов ударов.
    """
    players = list(player_stats.keys())
    
    # Собираем все типы ударов
    all_shot_types = set()
    for player in players:
        shot_types = player_stats[player].get('shot_types', {})
        all_shot_types.update(shot_types.keys())
    
    # Создаем данные для графика
    data = []
    for player in players:
        shot_types = player_stats[player].get('shot_types', {})
        total_shots = sum(shot_types.values()) if shot_types else 0
        
        for shot_type in all_shot_types:
            if total_shots > 0:
                percentage = shot_types.get(shot_type, 0) / total_shots * 100
            else:
                percentage = 0
                
            data.append({
                'Игрок': player,
                'Тип удара': shot_type,
                'Процент': percentage
            })
    
    df = pd.DataFrame(data)
    
    # Создаем график
    fig = px.bar(
        df, 
        x='Тип удара', 
        y='Процент', 
        color='Игрок',
        barmode='group',
        color_discrete_map={players[0]: colors['player1'], players[1]: colors['player2']} if len(players) > 1 else None,
        height=height
    )
    
    fig.update_layout(
        title='Распределение типов ударов',
        xaxis_title=None,
        yaxis_title='Процент (%)',
        legend_title='Игрок'
    )
    
    return fig

def create_serve_zones_chart(player_stats, player, color, height=400):
    """
    Создает тепловую карту зон подачи для игрока.
    """
    # Получаем данные о зонах подачи
    serve_zones = player_stats[player].get('serve_zones', {})
    total_serves = sum(serve_zones.values()) if serve_zones else 0
    
    # Стандартное теннисное поле (упрощенно)
    court_x = np.linspace(0, 1, 100)
    court_y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(court_x, court_y)
    Z = np.zeros_like(X)
    
    # Заполняем тепловую карту на основе данных о зонах
    # Здесь используется упрощенная модель, в реальном приложении нужно
    # соотносить названия зон с координатами на корте
    zone_to_coords = {
        "Wide": (0.2, 0.8),  # Широкая подача
        "Body": (0.5, 0.8),  # Подача в корпус
        "T": (0.8, 0.8),     # Подача по центральной линии
        "Center": (0.5, 0.5)  # Центр (для общих случаев)
    }
    
    for zone, count in serve_zones.items():
        if zone in zone_to_coords and total_serves > 0:
            x, y = zone_to_coords[zone]
            percentage = count / total_serves * 100
            
            # Добавляем "тепло" в тепловую карту
            for i in range(len(X)):
                for j in range(len(X[0])):
                    dist = np.sqrt((X[i, j] - x)**2 + (Y[i, j] - y)**2)
                    # Используем гауссову функцию для распределения "тепла"
                    Z[i, j] += percentage * np.exp(-10 * dist**2)
    
    # Создаем график
    fig = go.Figure()
    
    # Добавляем тепловую карту
    fig.add_trace(go.Heatmap(
        z=Z,
        colorscale=[[0, 'rgba(255,255,255,0)'], [1, color]],
        showscale=False
    ))
    
    # Добавляем разметку теннисного корта (упрощенно)
    # Внешние границы
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, line=dict(color="black"))
    # Центральная линия
    fig.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=1, line=dict(color="black"))
    
    # Настраиваем макет
    fig.update_layout(
        title=f"Зоны подачи - {player}",
        height=height,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_key_shots_chart(player_stats, player, color, height=400):
    """
    Создает график эффективности ключевых ударов игрока.
    """
    # Получаем данные о ключевых ударах
    key_shots = player_stats[player].get('key_shots', {})
    
    # Создаем данные для графика
    data = []
    for shot_type, stats in key_shots.items():
        if stats.get('total', 0) >= 2:  # Фильтруем удары с малым количеством наблюдений
            data.append({
                'Тип удара': shot_type,
                'Процент успешности': stats.get('win_percentage', 0),
                'Количество': stats.get('total', 0)
            })
    
    # Сортируем по количеству
    data = sorted(data, key=lambda x: x['Количество'], reverse=True)
    
    # Создаем DataFrame
    df = pd.DataFrame(data)
    
    # Если данных нет, возвращаем пустой график
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Эффективность ключевых ударов - {player}",
            height=height,
            xaxis_title="Нет данных",
            yaxis_title="Нет данных"
        )
        return fig
    
    # Создаем график
    fig = px.bar(
        df, 
        x='Тип удара', 
        y='Процент успешности',
        color_discrete_sequence=[color],
        text='Количество',
        height=height
    )
    
    fig.update_layout(
        title=f"Эффективность ключевых ударов - {player}",
        xaxis_title=None,
        yaxis_title='Процент успешности (%)',
        yaxis_range=[0, 100]
    )
    
    # Добавляем линию среднего значения
    avg = df['Процент успешности'].mean()
    fig.add_shape(
        type="line",
        xref="paper", yref="y",
        x0=0, y0=avg,
        x1=1, y1=avg,
        line=dict(
            color="red",
            width=2,
            dash="dash",
        )
    )
    
    fig.add_annotation(
        xref="paper", yref="y",
        x=0.01, y=avg,
        text=f"Среднее: {avg:.1f}%",
        showarrow=False,
        font=dict(color="red")
    )
    
    return fig

def create_shot_combinations_chart(player_stats, player, color, height=400, top_n=5):
    """
    Создает график эффективности комбинаций ударов игрока.
    """
    # Получаем данные о комбинациях ударов
    combinations = player_stats[player].get('shot_combinations', {})
    
    # Создаем данные для графика
    data = []
    for combo, stats in combinations.items():
        if stats.get('count', 0) >= 3:  # Фильтруем комбинации с малым количеством наблюдений
            data.append({
                'Комбинация': combo,
                'Процент успешности': stats.get('win_percentage', 0),
                'Количество': stats.get('count', 0)
            })
    
    # Сортируем по проценту успешности и берем top_n
    data = sorted(data, key=lambda x: x['Процент успешности'], reverse=True)[:top_n]
    
    # Создаем DataFrame
    df = pd.DataFrame(data)
    
    # Если данных нет, возвращаем пустой график
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Топ комбинации ударов - {player}",
            height=height,
            xaxis_title="Нет данных",
            yaxis_title="Нет данных"
        )
        return fig
    
    # Создаем график
    fig = px.bar(
        df, 
        y='Комбинация', 
        x='Процент успешности',
        color_discrete_sequence=[color],
        text='Количество',
        height=height,
        orientation='h'
    )
    
    fig.update_layout(
        title=f"Топ-{top_n} комбинаций ударов - {player}",
        xaxis_title='Процент успешности (%)',
        yaxis_title=None,
        xaxis_range=[0, 100]
    )
    
    return fig

def display_player_recommendations(recommendations, detail_level):
    """
    Отображает рекомендации для игрока.
    """
    if detail_level == "Минимальная":
        # Для минимальной детализации показываем только самое важное
        st.subheader("Ключевые рекомендации")
        
        # Объединяем все в один список
        all_recs = []
        for category, items in recommendations.items():
            all_recs.extend(items)
        
        # Выбираем максимум 3 самые важные рекомендации
        for rec in all_recs[:3]:
            st.write(f"• {rec}")
    else:
        # Для средней и подробной детализации показываем по категориям
        if recommendations['strengths']:
            st.subheader("💪 Сильные стороны")
            for strength in recommendations['strengths']:
                st.write(f"• {strength}")
        
        if recommendations['improvements']:
            st.subheader("🔄 Области для улучшения")
            for improvement in recommendations['improvements']:
                st.write(f"• {improvement}")
        
        if recommendations['tactics']:
            st.subheader("🎯 Тактические рекомендации")
            for tactic in recommendations['tactics']:
                st.write(f"• {tactic}")
        
        if recommendations['training_focus']:
            st.subheader("🏋️ Фокус тренировок")
            for focus in recommendations['training_focus']:
                st.write(f"• {focus}")
        
        if recommendations['mental_game']:
            st.subheader("🧠 Ментальный аспект")
            for mental in recommendations['mental_game']:
                st.write(f"• {mental}")

# Основная функция приложения
def main():
    st.title("Теннисная аналитика")
    
    # Добавляем настройки в сайдбар
    settings = add_settings_sidebar()
    color_scheme = get_color_scheme(settings)
    
    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите CSV файл с данными матча", type=['csv'])
    
    # Показываем подсказку, если включено
    if settings["show_help"]:
        st.info("""
        Загрузите CSV файл с данными теннисного матча. Файл должен содержать следующие столбцы:
        - Player_1: имя игрока, выполняющего действие
        - Serve: тип подачи ('1st', '2nd', '1st Serve', '2nd Serve')
        - Serve Zone: зона подачи (например, 'Wide', 'Body', 'T')
        - Serve Result: результат подачи ('In', 'In Play', 'Ace', 'Double Fault')
        - Shot Type: тип удара (например, 'Forehand', 'Backhand', 'Slice')
        - Finish Type: тип завершения розыгрыша ('Winner', 'Forced Error', 'Unforced Error')
        - Game Score: счет в гейме (например, '15-0', '30-15', '40-A')
        """)
    
    if uploaded_file is not None:
        try:
            # Чтение данных
            df = pd.read_csv(uploaded_file)
            
            # Проверка обязательных столбцов
            required_columns = ['Player_1', 'Serve', 'Shot Type']
            if not all(col in df.columns for col in required_columns):
                st.error("Загруженный файл не содержит необходимых столбцов для анализа")
                return
            
            # Анализ данных
            player_stats = analyze_match_data(df)
            players = list(player_stats.keys())
            
            if len(players) == 0:
                st.error("Не удалось найти информацию об игроках в данных")
                return
            
            # Показываем общую информацию
            st.header("Общая информация")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Игрок: {players[0]}")
                st.write(f"Эйсы: {player_stats[players[0]].get('aces', 0)}")
                st.write(f"Двойные ошибки: {player_stats[players[0]].get('double_faults', 0)}")
                st.write(f"Процент первой подачи: {player_stats[players[0]].get('first_serve_pct', 0)}%")
                st.write(f"Выигрыш на первой подаче: {player_stats[players[0]].get('first_serve_won_pct', 0)}%")
                st.write(f"Выигрыш на второй подаче: {player_stats[players[0]].get('second_serve_won_pct', 0)}%")
            
            if len(players) > 1:
                with col2:
                    st.subheader(f"Игрок: {players[1]}")
                    st.write(f"Эйсы: {player_stats[players[1]].get('aces', 0)}")
                    st.write(f"Двойные ошибки: {player_stats[players[1]].get('double_faults', 0)}")
                    st.write(f"Процент первой подачи: {player_stats[players[1]].get('first_serve_pct', 0)}%")
                    st.write(f"Выигрыш на первой подаче: {player_stats[players[1]].get('first_serve_won_pct', 0)}%")
                    st.write(f"Выигрыш на второй подаче: {player_stats[players[1]].get('second_serve_won_pct', 0)}%")
            
            # Визуализации
            st.header("Визуализация данных")
            
            # График статистики подачи
            st.plotly_chart(create_serve_stats_chart(player_stats, color_scheme, settings["chart_height"]), use_container_width=True)
            
            # График статистики розыгрышей
            st.plotly_chart(create_rally_stats_chart(player_stats, color_scheme, settings["chart_height"]), use_container_width=True)
            
            # График типов ударов
            st.plotly_chart(create_shot_types_chart(player_stats, color_scheme, settings["chart_height"]), use_container_width=True)
            
            # Зоны подачи и ключевые удары (в разных вкладках для каждого игрока)
            st.header("Детальная статистика игроков")
            
            tabs = st.tabs(players)
            for i, player in enumerate(players):
                with tabs[i]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(
                            create_serve_zones_chart(
                                player_stats, player, 
                                color_scheme['player1'] if i == 0 else color_scheme['player2'],
                                settings["chart_height"]
                            ),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.plotly_chart(
                            create_key_shots_chart(
                                player_stats, player, 
                                color_scheme['player1'] if i == 0 else color_scheme['player2'],
                                settings["chart_height"]
                            ),
                            use_container_width=True
                        )
                    
                    # Комбинации ударов
                    st.plotly_chart(
                        create_shot_combinations_chart(
                            player_stats, player, 
                            color_scheme['player1'] if i == 0 else color_scheme['player2'],
                            settings["chart_height"]
                        ),
                        use_container_width=True
                    )
            
            # Рекомендации
            st.header("Рекомендации для игроков")
            
            player_tabs = st.tabs(players)
            for i, player in enumerate(players):
                with player_tabs[i]:
                    # Генерируем рекомендации
                    opponent = players[1-i] if len(players) > 1 else None
                    opponent_stats = player_stats[opponent] if opponent else None
                    
                    recommendations = generate_player_recommendations(
                        player_stats[player], 
                        opponent_stats, 
                        settings["recommendation_detail"]
                    )
                    
                    # Отображаем рекомендации
                    display_player_recommendations(recommendations, settings["recommendation_detail"])
        
        except Exception as e:
            st.error(f"Произошла ошибка при анализе данных: {str(e)}")
            st.exception(e)

# Запуск приложения
if __name__ == "__main__":
    main()